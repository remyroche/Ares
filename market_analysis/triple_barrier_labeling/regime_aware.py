"""
Regime-Aware Triple Barrier Labeling

This module provides regime-aware triple barrier labeling functionality that adapts
barrier parameters based on market regimes detected by HMM or other regime detection methods.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

# Import common utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns
)
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation

# Import ML common utilities for regime detection
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.data_processing.regime_processing import RegimeProcessor

# Import core labeling functionality
from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod

# Setup logging
logger = logging.getLogger(__name__)

class RegimeType(Enum):
    """Types of market regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class RegimeAwareConfig:
    """Configuration for regime-aware labeling."""
    # Regime detection settings
    regime_detection_method: str = "hmm"  # "hmm", "volatility", "trend", "custom"
    regime_column: str = "regime"
    regime_transition_threshold: float = 0.1
    
    # Regime-specific barrier parameters
    regime_params: Dict[str, TripleBarrierConfig] = field(default_factory=dict)
    default_config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    
    # Adaptive parameters
    adaptive_parameters: bool = True
    parameter_smoothing: bool = True
    smoothing_window: int = 20
    
    # Regime transition handling
    handle_transitions: bool = True
    transition_buffer: int = 5
    
    # Quality settings
    min_samples_per_regime: int = 100
    regime_quality_threshold: float = 0.7

@dataclass
class RegimeStatistics:
    """Statistics for a specific regime."""
    regime: str
    sample_count: int
    label_distribution: Dict[str, float]
    profit_statistics: Dict[str, float]
    quality_score: float
    duration_seconds: float
    start_time: datetime
    end_time: datetime

class RegimeAwareLabeler:
    """Regime-aware triple barrier labeler."""
    
    def __init__(self, config: Optional[RegimeAwareConfig] = None):
        """Initialize the regime-aware labeler.
        
        Args:
            config: Configuration for regime-aware labeling
        """
        self.config = config or RegimeAwareConfig()
        self.logger = logging.getLogger(f"{__name__}.RegimeAwareLabeler")
        
        # Initialize components
        self._initialize_components()
        
        # Regime statistics
        self.regime_stats: Dict[str, RegimeStatistics] = {}
        
        self.logger.info("✅ RegimeAwareLabeler initialized successfully")

    def _initialize_components(self):
        """Initialize regime detection and labeling components."""
        self.logger.info("🔄 Initializing regime-aware components...")
        
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            
            # Initialize regime detection components
            if self.config.regime_detection_method == "hmm":
                self.regime_detector = HMMRegimeDetector()
            else:
                self.regime_detector = None
            
            # Initialize regime processor
            self.regime_processor = RegimeProcessor()
            
            # Initialize core labeler
            self.core_labeler = TripleBarrierLabeler()
            
            self.logger.info("✅ Regime-aware components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime-aware components: {e}")
            raise

    def create_regime_aware_labels(
        self,
        data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        config: Optional[RegimeAwareConfig] = None
    ) -> pd.DataFrame:
        """Create regime-aware triple barrier labels.
        
        Args:
            data: Market data with OHLC columns
            regime_data: Optional pre-computed regime data
            config: Optional configuration override
            
        Returns:
            DataFrame with regime-aware labels
        """
        config = config or self.config
        start_time = time.time()
        
        self.logger.info("🎯 Starting regime-aware triple barrier labeling")
        self.logger.info(f"📊 Input data shape: {data.shape}")
        
        try:
            # Detect regimes if not provided
            if regime_data is None:
                self.logger.info("🔍 Detecting market regimes...")
                regime_data = self._detect_regimes(data, config)
                self.logger.info(f"✅ Detected {len(regime_data['regime'].unique())} regimes")
            
            # Merge regime data with price data
            merged_data = data.merge(regime_data, left_index=True, right_index=True, how='left')
            
            # Get unique regimes
            regimes = merged_data['regime'].unique()
            self.logger.info(f"📈 Processing {len(regimes)} regimes: {regimes}")
            
            # Process each regime
            regime_results = []
            regime_stats = {}
            
            for regime in regimes:
                regime_start_time = time.time()
                
                # Get regime-specific configuration
                regime_config = self._get_regime_config(regime, config)
                
                # Filter data for this regime
                regime_mask = merged_data['regime'] == regime
                regime_data_subset = merged_data[regime_mask]
                
                self.logger.debug(f"🔄 Processing regime {regime} with {len(regime_data_subset)} samples")
                
                # Create labels for this regime
                regime_labels = self.core_labeler._create_standard_triple_barrier(
                    regime_data_subset, regime_config
                )
                regime_labels['regime'] = regime
                regime_labels['regime_config_pt'] = regime_config.pt_mult
                regime_labels['regime_config_sl'] = regime_config.sl_mult
                
                # Calculate regime statistics
                regime_stat = self._calculate_regime_statistics(
                    regime_labels, regime, regime_start_time
                )
                regime_stats[regime] = regime_stat
                
                regime_results.append(regime_labels)
                
                regime_time = time.time() - regime_start_time
                self.logger.debug(f"✅ Regime {regime} completed in {regime_time:.3f}s")
            
            # Combine results
            result = pd.concat(regime_results, ignore_index=True)
            result = result.sort_index()
            result['barrier_type'] = 'regime_aware_triple_barrier'
            
            # Store regime statistics
            self.regime_stats = regime_stats
            
            # Log summary
            total_time = time.time() - start_time
            self.logger.info(f"✅ Regime-aware labeling completed in {total_time:.3f}s")
            self._log_regime_summary(regime_stats)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"❌ Failed to create regime-aware labels after {total_time:.3f}s: {e}")
            raise

    def _detect_regimes(self, data: pd.DataFrame, config: RegimeAwareConfig) -> pd.DataFrame:
        """Detect market regimes using specified method."""
        self.logger.debug(f"🔍 Detecting regimes using method: {config.regime_detection_method}")
        
        if config.regime_detection_method == "hmm":
            return self._detect_regimes_hmm(data, config)
        elif config.regime_detection_method == "volatility":
            return self._detect_regimes_volatility(data, config)
        elif config.regime_detection_method == "trend":
            return self._detect_regimes_trend(data, config)
        else:
            raise ValueError(f"Unsupported regime detection method: {config.regime_detection_method}")

    def _detect_regimes_hmm(self, data: pd.DataFrame, config: RegimeAwareConfig) -> pd.DataFrame:
        """Detect regimes using HMM."""
        try:
            if self.regime_detector is None:
                raise ValueError("HMM regime detector not initialized")
            
            # Prepare features for HMM
            features = self._prepare_hmm_features(data)
            
            # Detect regimes
            regimes = self.regime_detector.detect_regimes(features)
            
            # Create regime DataFrame
            regime_df = pd.DataFrame({
                'regime': regimes
            }, index=data.index)
            
            return regime_df
            
        except Exception as e:
            self.logger.error(f"❌ HMM regime detection failed: {e}")
            # Fallback to volatility-based detection
            return self._detect_regimes_volatility(data, config)

    def _detect_regimes_volatility(self, data: pd.DataFrame, config: RegimeAwareConfig) -> pd.DataFrame:
        """Detect regimes based on volatility."""
        try:
            # Calculate rolling volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std()
            
            # Define volatility thresholds
            vol_high = volatility.quantile(0.8)
            vol_low = volatility.quantile(0.2)
            
            # Classify regimes
            regimes = []
            for vol in volatility:
                if pd.isna(vol):
                    regimes.append(RegimeType.SIDEWAYS.value)
                elif vol > vol_high:
                    regimes.append(RegimeType.HIGH_VOLATILITY.value)
                elif vol < vol_low:
                    regimes.append(RegimeType.LOW_VOLATILITY.value)
                else:
                    regimes.append(RegimeType.SIDEWAYS.value)
            
            # Create regime DataFrame
            regime_df = pd.DataFrame({
                'regime': regimes
            }, index=data.index)
            
            return regime_df
            
        except Exception as e:
            self.logger.error(f"❌ Volatility regime detection failed: {e}")
            # Fallback to simple regime assignment
            return self._create_default_regimes(data)

    def _detect_regimes_trend(self, data: pd.DataFrame, config: RegimeAwareConfig) -> pd.DataFrame:
        """Detect regimes based on trend analysis."""
        try:
            # Calculate trend indicators
            sma_short = data['close'].rolling(window=10).mean()
            sma_long = data['close'].rolling(window=30).mean()
            
            # Classify regimes based on trend
            regimes = []
            for i in range(len(data)):
                if i < 30:  # Not enough data for trend analysis
                    regimes.append(RegimeType.SIDEWAYS.value)
                else:
                    short_ma = sma_short.iloc[i]
                    long_ma = sma_long.iloc[i]
                    
                    if short_ma > long_ma * 1.02:  # 2% above long MA
                        regimes.append(RegimeType.BULL_MARKET.value)
                    elif short_ma < long_ma * 0.98:  # 2% below long MA
                        regimes.append(RegimeType.BEAR_MARKET.value)
                    else:
                        regimes.append(RegimeType.SIDEWAYS.value)
            
            # Create regime DataFrame
            regime_df = pd.DataFrame({
                'regime': regimes
            }, index=data.index)
            
            return regime_df
            
        except Exception as e:
            self.logger.error(f"❌ Trend regime detection failed: {e}")
            # Fallback to simple regime assignment
            return self._create_default_regimes(data)

    def _create_default_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create default regime assignment when detection fails."""
        self.logger.warning("⚠️ Using default regime assignment")
        
        # Simple alternating regime assignment
        regimes = []
        for i in range(len(data)):
            if i % 100 < 50:  # Alternate every 100 periods
                regimes.append(RegimeType.BULL_MARKET.value)
            else:
                regimes.append(RegimeType.BEAR_MARKET.value)
        
        regime_df = pd.DataFrame({
            'regime': regimes
        }, index=data.index)
        
        return regime_df

    def _prepare_hmm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for HMM regime detection."""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['macd'] = self._calculate_macd(data['close'])
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ma'] = data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _get_regime_config(self, regime: str, config: RegimeAwareConfig) -> TripleBarrierConfig:
        """Get regime-specific configuration."""
        if regime in config.regime_params:
            return config.regime_params[regime]
        
        # Create adaptive configuration based on regime type
        base_config = config.default_config
        
        if regime == RegimeType.BULL_MARKET.value:
            # More aggressive profit targets in bull markets
            regime_config = TripleBarrierConfig(
                pt_mult=base_config.pt_mult * 1.2,
                sl_mult=base_config.sl_mult * 0.8,
                min_holding_period=base_config.min_holding_period,
                max_holding_period=base_config.max_holding_period,
                transaction_cost=base_config.transaction_cost
            )
        elif regime == RegimeType.BEAR_MARKET.value:
            # More conservative in bear markets
            regime_config = TripleBarrierConfig(
                pt_mult=base_config.pt_mult * 0.8,
                sl_mult=base_config.sl_mult * 1.2,
                min_holding_period=base_config.min_holding_period,
                max_holding_period=base_config.max_holding_period,
                transaction_cost=base_config.transaction_cost
            )
        elif regime == RegimeType.HIGH_VOLATILITY.value:
            # Wider barriers in high volatility
            regime_config = TripleBarrierConfig(
                pt_mult=base_config.pt_mult * 1.5,
                sl_mult=base_config.sl_mult * 1.5,
                min_holding_period=base_config.min_holding_period,
                max_holding_period=base_config.max_holding_period,
                transaction_cost=base_config.transaction_cost
            )
        elif regime == RegimeType.LOW_VOLATILITY.value:
            # Tighter barriers in low volatility
            regime_config = TripleBarrierConfig(
                pt_mult=base_config.pt_mult * 0.7,
                sl_mult=base_config.sl_mult * 0.7,
                min_holding_period=base_config.min_holding_period,
                max_holding_period=base_config.max_holding_period,
                transaction_cost=base_config.transaction_cost
            )
        else:
            # Default configuration for other regimes
            regime_config = base_config
        
        return regime_config

    def _calculate_regime_statistics(
        self, 
        regime_labels: pd.DataFrame, 
        regime: str, 
        start_time: float
    ) -> RegimeStatistics:
        """Calculate statistics for a specific regime."""
        try:
            # Label distribution
            label_counts = regime_labels['label'].value_counts()
            total_labels = len(regime_labels)
            label_distribution = {
                'positive': safe_divide(label_counts.get(1, 0), total_labels),
                'negative': safe_divide(label_counts.get(-1, 0), total_labels),
                'neutral': safe_divide(label_counts.get(0, 0), total_labels)
            }
            
            # Profit statistics
            profits = regime_labels['profit_pct'].values
            profit_statistics = {
                'mean': np.mean(profits),
                'std': np.std(profits),
                'min': np.min(profits),
                'max': np.max(profits),
                'positive_ratio': safe_divide(np.sum(profits > 0), len(profits))
            }
            
            # Quality score
            quality_score = self._calculate_regime_quality_score(
                label_distribution, profit_statistics
            )
            
            # Duration
            duration_seconds = time.time() - start_time
            
            # Time range
            start_time_dt = regime_labels.index.min()
            end_time_dt = regime_labels.index.max()
            
            return RegimeStatistics(
                regime=regime,
                sample_count=total_labels,
                label_distribution=label_distribution,
                profit_statistics=profit_statistics,
                quality_score=quality_score,
                duration_seconds=duration_seconds,
                start_time=start_time_dt,
                end_time=end_time_dt
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate regime statistics for {regime}: {e}")
            return RegimeStatistics(
                regime=regime,
                sample_count=0,
                label_distribution={},
                profit_statistics={},
                quality_score=0.0,
                duration_seconds=0.0,
                start_time=datetime.now(),
                end_time=datetime.now()
            )

    def _calculate_regime_quality_score(
        self, 
        label_distribution: Dict[str, float], 
        profit_statistics: Dict[str, float]
    ) -> float:
        """Calculate quality score for a regime."""
        try:
            # Balance score (prefer balanced labels)
            balance_score = 1.0 - abs(
                label_distribution.get('positive', 0) - label_distribution.get('negative', 0)
            )
            
            # Profit consistency score
            profit_consistency = 1.0 - min(1.0, profit_statistics.get('std', 0) / max(0.001, abs(profit_statistics.get('mean', 0))))
            
            # Overall quality
            quality_score = (balance_score * 0.5 + profit_consistency * 0.5)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0

    def _log_regime_summary(self, regime_stats: Dict[str, RegimeStatistics]):
        """Log summary of regime statistics."""
        self.logger.info("📊 Regime Summary:")
        
        for regime, stats in regime_stats.items():
            self.logger.info(f"  📈 {regime}:")
            self.logger.info(f"    📊 Samples: {stats.sample_count}")
            self.logger.info(f"    🎯 Quality: {stats.quality_score:.3f}")
            self.logger.info(f"    📈 Label dist: {stats.label_distribution}")
            self.logger.info(f"    💰 Profit stats: {stats.profit_statistics}")

    def get_regime_statistics(self) -> Dict[str, RegimeStatistics]:
        """Get regime statistics."""
        return self.regime_stats.copy()

    def analyze_regime_transitions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime transitions in the data."""
        try:
            if 'regime' not in data.columns:
                return {'error': 'No regime column found in data'}
            
            regimes = data['regime'].values
            transitions = []
            
            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i-1]:
                    transitions.append({
                        'from_regime': regimes[i-1],
                        'to_regime': regimes[i],
                        'timestamp': data.index[i],
                        'index': i
                    })
            
            # Calculate transition statistics
            transition_counts = {}
            for trans in transitions:
                key = f"{trans['from_regime']} -> {trans['to_regime']}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            return {
                'total_transitions': len(transitions),
                'transition_counts': transition_counts,
                'transitions': transitions,
                'regime_duration_stats': self._calculate_regime_durations(data)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze regime transitions: {e}")
            return {'error': str(e)}

    def _calculate_regime_durations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate duration statistics for each regime."""
        try:
            if 'regime' not in data.columns:
                return {}
            
            regimes = data['regime'].values
            regime_durations = {}
            current_regime = regimes[0]
            current_duration = 1
            
            for i in range(1, len(regimes)):
                if regimes[i] == current_regime:
                    current_duration += 1
                else:
                    # Record duration for previous regime
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    regime_durations[current_regime].append(current_duration)
                    
                    # Start new regime
                    current_regime = regimes[i]
                    current_duration = 1
            
            # Add last regime duration
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(current_duration)
            
            # Calculate statistics
            duration_stats = {}
            for regime, durations in regime_durations.items():
                duration_stats[regime] = {
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'total_periods': len(durations)
                }
            
            return duration_stats
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate regime durations: {e}")
            return {}