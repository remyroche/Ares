"""
Enhanced SR Feature Extractor with Historical Integration

This module extends the basic SR feature extractor to include comprehensive
historical SR level analysis for ML learning and trading applications.

Key Features:
- Historical SR level loading and analysis
- Level persistence and evolution tracking
- Historical touch pattern analysis
- Bounce success rate calculations
- ML-ready feature extraction
- Trading-relevant probability features
- Risk assessment based on historical data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

# Import base SR feature extractor
from .sr_feature_extractor import SRFeatureExtractor, SRFeatureConfig, get_sr_feature_extractor

# Import math validation utilities
from .math_validation import safe_divide, validate_positive

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

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

@dataclass
class HistoricalSRConfig:
    """Configuration for historical SR level analysis."""
    # Historical data settings
    load_historical_levels: bool = True
    historical_data_path: str = "sr_levels_history.json"
    current_levels_path: str = "sr_levels.json"

    # Historical analysis settings
    max_history_days: int = 30  # Maximum days of history to analyze
    min_level_age_hours: float = 1.0  # Minimum age for level analysis
    max_level_age_hours: float = 720.0  # Maximum age for level analysis (30 days)

    # Feature extraction settings
    enable_level_persistence_features: bool = True
    enable_historical_touch_analysis: bool = True
    enable_bounce_success_analysis: bool = True
    enable_level_evolution_features: bool = True
    enable_ml_ready_features: bool = True
    enable_trading_features: bool = True

    # ML feature settings
    create_feature_vectors: bool = True
    normalize_features: bool = True
    create_interaction_features: bool = True

    # Trading feature settings
    calculate_reliability_scores: bool = True
    calculate_probability_features: bool = True
    calculate_risk_assessment: bool = True
    calculate_timing_features: bool = True

class HistoricalSRAnalyzer:
    """Analyzer for historical SR level data."""

    def __init__(self, config: Optional[HistoricalSRConfig] = None):
        self.config = config or HistoricalSRConfig()
        self.logger = logger.getChild('HistoricalSRAnalyzer')

        # Historical data storage
        self.historical_levels = []
        self.current_levels = {}
        self.level_evolution_data = {}
        self.touch_history = {}
        self.bounce_history = {}

        # Load historical data
        if self.config.load_historical_levels:
            self._load_historical_data()

        self.logger.info("🚀 Historical SR Analyzer initialized")
        self.logger.info(f"   Historical levels loaded: {len(self.historical_levels)}")
        self.logger.info(f"   Current levels loaded: {len(self.current_levels.get('support_levels', [])) + len(self.current_levels.get('resistance_levels', []))}")

    def _load_historical_data(self):
        """Load historical SR level data from JSON files."""
        try:
            # Load current levels
            current_path = Path(self.config.current_levels_path)
            if current_path.exists():
                with open(current_path, 'r') as f:
                    self.current_levels = json.load(f)
                self.logger.info(f"✅ Loaded current SR levels from {current_path}")
            else:
                self.logger.warning(f"⚠️ Current SR levels file not found: {current_path}")

            # Load historical levels
            historical_path = Path(self.config.historical_data_path)
            if historical_path.exists():
                with open(historical_path, 'r') as f:
                    self.historical_levels = json.load(f)
                self.logger.info(f"✅ Loaded {len(self.historical_levels)} historical SR level snapshots")

                # Process historical data
                self._process_historical_data()
            else:
                self.logger.warning(f"⚠️ Historical SR levels file not found: {historical_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load historical data: {e}")
            self.historical_levels = []
            self.current_levels = {}

    def _process_historical_data(self):
        """Process historical data to extract evolution patterns."""
        try:
            if not self.historical_levels:
                return

            # Sort by timestamp
            self.historical_levels.sort(key=lambda x: x['timestamp'])

            # Extract level evolution data
            self._extract_level_evolution()

            # Extract touch history
            self._extract_touch_history()

            # Extract bounce history
            self._extract_bounce_history()

            self.logger.info("✅ Historical data processing completed")

        except Exception as e:
            self.logger.error(f"❌ Failed to process historical data: {e}")

    def _extract_level_evolution(self):
        """Extract level evolution patterns from historical data."""
        level_evolution = {}

        for snapshot in self.historical_levels:
            timestamp = snapshot['timestamp']
            data = snapshot['data']

            # Process support levels
            for level in data.get('support_levels', []):
                price = level['price']
                level_key = f"support_{price:.6f}"

                if level_key not in level_evolution:
                    level_evolution[level_key] = {
                        'price': price,
                        'level_type': 'support',
                        'creation_time': level.get('creation_time', timestamp),
                        'evolution': [],
                        'total_touches': 0,
                        'total_bounces': 0,
                        'strength_history': [],
                        'volume_history': []
                    }

                # Add evolution point
                level_evolution[level_key]['evolution'].append({
                    'timestamp': timestamp,
                    'strength': level.get('strength', 0.0),
                    'volume': level.get('volume', 0.0),
                    'touch_count': level.get('touch_count', 0),
                    'age_hours': level.get('age_hours', 0.0),
                    'bounce_rate': level.get('bounce_rate', 0.0)
                })

                # Update totals
                level_evolution[level_key]['total_touches'] += level.get('touch_count', 0)
                level_evolution[level_key]['total_bounces'] += level.get('bounce_count', 0)
                level_evolution[level_key]['strength_history'].append(level.get('strength', 0.0))
                level_evolution[level_key]['volume_history'].append(level.get('volume', 0.0))

            # Process resistance levels
            for level in data.get('resistance_levels', []):
                price = level['price']
                level_key = f"resistance_{price:.6f}"

                if level_key not in level_evolution:
                    level_evolution[level_key] = {
                        'price': price,
                        'level_type': 'resistance',
                        'creation_time': level.get('creation_time', timestamp),
                        'evolution': [],
                        'total_touches': 0,
                        'total_bounces': 0,
                        'strength_history': [],
                        'volume_history': []
                    }

                # Add evolution point
                level_evolution[level_key]['evolution'].append({
                    'timestamp': timestamp,
                    'strength': level.get('strength', 0.0),
                    'volume': level.get('volume', 0.0),
                    'touch_count': level.get('touch_count', 0),
                    'age_hours': level.get('age_hours', 0.0),
                    'bounce_rate': level.get('bounce_rate', 0.0)
                })

                # Update totals
                level_evolution[level_key]['total_touches'] += level.get('touch_count', 0)
                level_evolution[level_key]['total_bounces'] += level.get('bounce_count', 0)
                level_evolution[level_key]['strength_history'].append(level.get('strength', 0.0))
                level_evolution[level_key]['volume_history'].append(level.get('volume', 0.0))

        self.level_evolution_data = level_evolution
        self.logger.info(f"✅ Extracted evolution data for {len(level_evolution)} levels")

    def _extract_touch_history(self):
        """Extract touch history patterns."""
        touch_history = {}

        for level_key, evolution_data in self.level_evolution_data.items():
            price = evolution_data['price']
            level_type = evolution_data['level_type']

            # Calculate touch frequency over time
            touch_counts = [point['touch_count'] for point in evolution_data['evolution']]
            timestamps = [point['timestamp'] for point in evolution_data['evolution']]

            if len(touch_counts) > 1:
                # Calculate touch frequency (touches per hour)
                time_diffs = []
                for i in range(1, len(timestamps)):
                    try:
                        t1 = datetime.fromisoformat(timestamps[i-1].replace('Z', '+00:00'))
                        t2 = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
                        diff_hours = (t2 - t1).total_seconds() / 3600
                        if diff_hours > 0:
                            time_diffs.append(diff_hours)
                    except:
                        continue

                if time_diffs:
                    avg_time_diff = np.mean(time_diffs)
                    touch_frequency = np.mean(touch_counts) / max(avg_time_diff, 1.0)
                else:
                    touch_frequency = 0.0
            else:
                touch_frequency = 0.0

            touch_history[level_key] = {
                'price': price,
                'level_type': level_type,
                'total_touches': evolution_data['total_touches'],
                'touch_frequency': touch_frequency,
                'touch_counts': touch_counts,
                'timestamps': timestamps,
                'avg_touches_per_snapshot': np.mean(touch_counts) if touch_counts else 0.0,
                'max_touches': max(touch_counts) if touch_counts else 0.0,
                'min_touches': min(touch_counts) if touch_counts else 0.0
            }

        self.touch_history = touch_history
        self.logger.info(f"✅ Extracted touch history for {len(touch_history)} levels")

    def _extract_bounce_history(self):
        """Extract bounce history patterns."""
        bounce_history = {}

        for level_key, evolution_data in self.level_evolution_data.items():
            price = evolution_data['price']
            level_type = evolution_data['level_type']

            # Calculate bounce success rate
            bounce_rates = [point['bounce_rate'] for point in evolution_data['evolution']]
            strength_history = evolution_data['strength_history']

            if bounce_rates:
                avg_bounce_rate = np.mean(bounce_rates)
                max_bounce_rate = max(bounce_rates)
                min_bounce_rate = min(bounce_rates)
                bounce_consistency = 1.0 - np.std(bounce_rates) if len(bounce_rates) > 1 else 1.0
            else:
                avg_bounce_rate = 0.0
                max_bounce_rate = 0.0
                min_bounce_rate = 0.0
                bounce_consistency = 0.0

            # Calculate strength evolution
            if strength_history:
                strength_trend = np.polyfit(range(len(strength_history)), strength_history, 1)[0] if len(strength_history) > 1 else 0.0
                strength_volatility = np.std(strength_history) if len(strength_history) > 1 else 0.0
                current_strength = strength_history[-1] if strength_history else 0.0
            else:
                strength_trend = 0.0
                strength_volatility = 0.0
                current_strength = 0.0

            bounce_history[level_key] = {
                'price': price,
                'level_type': level_type,
                'avg_bounce_rate': avg_bounce_rate,
                'max_bounce_rate': max_bounce_rate,
                'min_bounce_rate': min_bounce_rate,
                'bounce_consistency': bounce_consistency,
                'strength_trend': strength_trend,
                'strength_volatility': strength_volatility,
                'current_strength': current_strength,
                'total_bounces': evolution_data['total_bounces']
            }

        self.bounce_history = bounce_history
        self.logger.info(f"✅ Extracted bounce history for {len(bounce_history)} levels")

    def get_level_reliability_score(self, price: float, level_type: str) -> float:
        """Calculate reliability score for a level based on historical data."""
        level_key = f"{level_type}_{price:.6f}"

        if level_key not in self.level_evolution_data:
            return 0.5  # Default reliability for unknown levels

        evolution_data = self.level_evolution_data[level_key]
        touch_data = self.touch_history.get(level_key, {})
        bounce_data = self.bounce_history.get(level_key, {})

        # Calculate reliability based on multiple factors
        factors = []

        # Factor 1: Level age (older levels are more reliable)
        if evolution_data['evolution']:
            latest_age = evolution_data['evolution'][-1]['age_hours']
            age_score = min(latest_age / 24.0, 1.0)  # Normalize to 1.0 for 24+ hours
            factors.append(age_score * 0.2)

        # Factor 2: Touch frequency (more touches = more reliable)
        touch_frequency = touch_data.get('touch_frequency', 0.0)
        touch_score = min(touch_frequency / 10.0, 1.0)  # Normalize to 1.0 for 10+ touches/hour
        factors.append(touch_score * 0.3)

        # Factor 3: Bounce success rate
        bounce_rate = bounce_data.get('avg_bounce_rate', 0.0)
        factors.append(bounce_rate * 0.3)

        # Factor 4: Strength consistency
        strength_volatility = bounce_data.get('strength_volatility', 1.0)
        consistency_score = max(0.0, 1.0 - strength_volatility)
        factors.append(consistency_score * 0.2)

        # Calculate weighted reliability score
        reliability_score = sum(factors) if factors else 0.5
        return min(max(reliability_score, 0.0), 1.0)

    def get_level_probability_features(self, price: float, level_type: str) -> Dict[str, float]:
        """Get probability features for a level based on historical data."""
        level_key = f"{level_type}_{price:.6f}"

        if level_key not in self.level_evolution_data:
            return {
                'bounce_probability': 0.5,
                'breakout_probability': 0.5,
                'touch_probability': 0.5,
                'strength_probability': 0.5
            }

        touch_data = self.touch_history.get(level_key, {})
        bounce_data = self.bounce_history.get(level_key, {})

        # Calculate probabilities based on historical data
        bounce_probability = bounce_data.get('avg_bounce_rate', 0.5)
        breakout_probability = 1.0 - bounce_probability

        # Touch probability based on frequency
        touch_frequency = touch_data.get('touch_frequency', 0.0)
        touch_probability = min(touch_frequency / 5.0, 1.0)  # Normalize to 1.0 for 5+ touches/hour

        # Strength probability based on current strength
        current_strength = bounce_data.get('current_strength', 0.5)
        strength_probability = current_strength

        return {
            'bounce_probability': bounce_probability,
            'breakout_probability': breakout_probability,
            'touch_probability': touch_probability,
            'strength_probability': strength_probability
        }

class EnhancedSRFeatureExtractor(SRFeatureExtractor):
    """Enhanced SR feature extractor with historical integration."""

    def __init__(self, config: Optional[SRFeatureConfig] = None,
                 historical_config: Optional[HistoricalSRConfig] = None):
        super().__init__(config)

        self.historical_config = historical_config or HistoricalSRConfig()
        self.historical_analyzer = HistoricalSRAnalyzer(self.historical_config)

        self.logger.info("🚀 Enhanced SR Feature Extractor initialized with historical integration")

    def extract_historical_sr_features(self, data: pd.DataFrame,
                                     sr_levels: Optional[Dict[str, Any]] = None,
                                     regime_labels: Optional[pd.Series] = None) -> pd.DataFrame:
        """Extract SR features with historical integration."""
        try:
            self.logger.info(f"🔧 Extracting enhanced SR features with historical integration from {len(data)} rows")
            start_time = time.time()

            # Get base SR features
            base_features = self.extract_sr_features(data, sr_levels, regime_labels)

            # Initialize enhanced features DataFrame
            enhanced_features = base_features.copy()

            # Add historical features
            if self.historical_config.enable_level_persistence_features:
                historical_features = self._extract_level_persistence_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, historical_features], axis=1)

            if self.historical_config.enable_historical_touch_analysis:
                touch_features = self._extract_historical_touch_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, touch_features], axis=1)

            if self.historical_config.enable_bounce_success_analysis:
                bounce_features = self._extract_bounce_success_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, bounce_features], axis=1)

            if self.historical_config.enable_level_evolution_features:
                evolution_features = self._extract_level_evolution_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, evolution_features], axis=1)

            if self.historical_config.enable_ml_ready_features:
                ml_features = self._extract_ml_ready_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, ml_features], axis=1)

            if self.historical_config.enable_trading_features:
                trading_features = self._extract_trading_features(data, sr_levels)
                enhanced_features = pd.concat([enhanced_features, trading_features], axis=1)

            # Clean and validate features
            enhanced_features = self._clean_sr_features(enhanced_features)

            processing_time = time.time() - start_time
            self.logger.info(f"✅ Enhanced SR feature extraction completed in {processing_time:.2f}s")
            self.logger.info(f"   Base features: {base_features.shape[1]}")
            self.logger.info(f"   Enhanced features: {enhanced_features.shape[1]}")
            self.logger.info(f"   Historical features added: {enhanced_features.shape[1] - base_features.shape[1]}")

            return enhanced_features

        except Exception as e:
            self.logger.error(f"❌ Enhanced SR feature extraction failed: {e}")
            # Fallback to base features
            return self.extract_sr_features(data, sr_levels, regime_labels)

    def _extract_level_persistence_features(self, data: pd.DataFrame,
                                          sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract level persistence features."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Calculate persistence features for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Support level persistence
                support_persistence = []
                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        reliability = self.historical_analyzer.get_level_reliability_score(level_price, 'support')
                        support_persistence.append(reliability)

                # Resistance level persistence
                resistance_persistence = []
                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        reliability = self.historical_analyzer.get_level_reliability_score(level_price, 'resistance')
                        resistance_persistence.append(reliability)

                # Store features
                features.loc[data.index[i], 'avg_support_persistence'] = np.mean(support_persistence) if support_persistence else 0.0
                features.loc[data.index[i], 'max_support_persistence'] = max(support_persistence) if support_persistence else 0.0
                features.loc[data.index[i], 'avg_resistance_persistence'] = np.mean(resistance_persistence) if resistance_persistence else 0.0
                features.loc[data.index[i], 'max_resistance_persistence'] = max(resistance_persistence) if resistance_persistence else 0.0
                features.loc[data.index[i], 'total_persistence_score'] = np.mean(support_persistence + resistance_persistence) if (support_persistence or resistance_persistence) else 0.0

        return features

    def _extract_historical_touch_features(self, data: pd.DataFrame,
                                         sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract historical touch analysis features."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Calculate touch features for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Support level touch features
                support_touch_freq = []
                support_total_touches = []
                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"support_{level_price:.6f}"
                        touch_data = self.historical_analyzer.touch_history.get(level_key, {})
                        support_touch_freq.append(touch_data.get('touch_frequency', 0.0))
                        support_total_touches.append(touch_data.get('total_touches', 0))

                # Resistance level touch features
                resistance_touch_freq = []
                resistance_total_touches = []
                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"resistance_{level_price:.6f}"
                        touch_data = self.historical_analyzer.touch_history.get(level_key, {})
                        resistance_touch_freq.append(touch_data.get('touch_frequency', 0.0))
                        resistance_total_touches.append(touch_data.get('total_touches', 0))

                # Store features
                features.loc[data.index[i], 'avg_support_touch_frequency'] = np.mean(support_touch_freq) if support_touch_freq else 0.0
                features.loc[data.index[i], 'total_support_touches'] = sum(support_total_touches)
                features.loc[data.index[i], 'avg_resistance_touch_frequency'] = np.mean(resistance_touch_freq) if resistance_touch_freq else 0.0
                features.loc[data.index[i], 'total_resistance_touches'] = sum(resistance_total_touches)

        return features

    def _extract_bounce_success_features(self, data: pd.DataFrame,
                                       sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract bounce success analysis features."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Calculate bounce features for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Support level bounce features
                support_bounce_rates = []
                support_bounce_consistency = []
                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"support_{level_price:.6f}"
                        bounce_data = self.historical_analyzer.bounce_history.get(level_key, {})
                        support_bounce_rates.append(bounce_data.get('avg_bounce_rate', 0.0))
                        support_bounce_consistency.append(bounce_data.get('bounce_consistency', 0.0))

                # Resistance level bounce features
                resistance_bounce_rates = []
                resistance_bounce_consistency = []
                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"resistance_{level_price:.6f}"
                        bounce_data = self.historical_analyzer.bounce_history.get(level_key, {})
                        resistance_bounce_rates.append(bounce_data.get('avg_bounce_rate', 0.0))
                        resistance_bounce_consistency.append(bounce_data.get('bounce_consistency', 0.0))

                # Store features
                features.loc[data.index[i], 'avg_support_bounce_rate'] = np.mean(support_bounce_rates) if support_bounce_rates else 0.0
                features.loc[data.index[i], 'avg_support_bounce_consistency'] = np.mean(support_bounce_consistency) if support_bounce_consistency else 0.0
                features.loc[data.index[i], 'avg_resistance_bounce_rate'] = np.mean(resistance_bounce_rates) if resistance_bounce_rates else 0.0
                features.loc[data.index[i], 'avg_resistance_bounce_consistency'] = np.mean(resistance_bounce_consistency) if resistance_bounce_consistency else 0.0

        return features

    def _extract_level_evolution_features(self, data: pd.DataFrame,
                                        sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract level evolution features."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Calculate evolution features for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Support level evolution features
                support_strength_trends = []
                support_strength_volatility = []
                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"support_{level_price:.6f}"
                        bounce_data = self.historical_analyzer.bounce_history.get(level_key, {})
                        support_strength_trends.append(bounce_data.get('strength_trend', 0.0))
                        support_strength_volatility.append(bounce_data.get('strength_volatility', 0.0))

                # Resistance level evolution features
                resistance_strength_trends = []
                resistance_strength_volatility = []
                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.01:  # Within 1%
                        level_key = f"resistance_{level_price:.6f}"
                        bounce_data = self.historical_analyzer.bounce_history.get(level_key, {})
                        resistance_strength_trends.append(bounce_data.get('strength_trend', 0.0))
                        resistance_strength_volatility.append(bounce_data.get('strength_volatility', 0.0))

                # Store features
                features.loc[data.index[i], 'avg_support_strength_trend'] = np.mean(support_strength_trends) if support_strength_trends else 0.0
                features.loc[data.index[i], 'avg_support_strength_volatility'] = np.mean(support_strength_volatility) if support_strength_volatility else 0.0
                features.loc[data.index[i], 'avg_resistance_strength_trend'] = np.mean(resistance_strength_trends) if resistance_strength_trends else 0.0
                features.loc[data.index[i], 'avg_resistance_strength_volatility'] = np.mean(resistance_strength_volatility) if resistance_strength_volatility else 0.0

        return features

    def _extract_ml_ready_features(self, data: pd.DataFrame,
                                 sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract ML-ready feature vectors."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Create ML-ready feature vectors for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Create feature vector for nearby levels
                nearby_levels = []

                # Add support levels
                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.02:  # Within 2%
                        level_key = f"support_{level_price:.6f}"
                        reliability = self.historical_analyzer.get_level_reliability_score(level_price, 'support')
                        probabilities = self.historical_analyzer.get_level_probability_features(level_price, 'support')

                        nearby_levels.append({
                            'price': level_price,
                            'type': 'support',
                            'distance': distance,
                            'reliability': reliability,
                            'bounce_probability': probabilities['bounce_probability'],
                            'breakout_probability': probabilities['breakout_probability'],
                            'touch_probability': probabilities['touch_probability'],
                            'strength_probability': probabilities['strength_probability']
                        })

                # Add resistance levels
                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance <= 0.02:  # Within 2%
                        level_key = f"resistance_{level_price:.6f}"
                        reliability = self.historical_analyzer.get_level_reliability_score(level_price, 'resistance')
                        probabilities = self.historical_analyzer.get_level_probability_features(level_price, 'resistance')

                        nearby_levels.append({
                            'price': level_price,
                            'type': 'resistance',
                            'distance': distance,
                            'reliability': reliability,
                            'bounce_probability': probabilities['bounce_probability'],
                            'breakout_probability': probabilities['breakout_probability'],
                            'touch_probability': probabilities['touch_probability'],
                            'strength_probability': probabilities['strength_probability']
                        })

                # Calculate aggregate ML features
                if nearby_levels:
                    features.loc[data.index[i], 'ml_avg_reliability'] = np.mean([l['reliability'] for l in nearby_levels])
                    features.loc[data.index[i], 'ml_max_reliability'] = max([l['reliability'] for l in nearby_levels])
                    features.loc[data.index[i], 'ml_avg_bounce_probability'] = np.mean([l['bounce_probability'] for l in nearby_levels])
                    features.loc[data.index[i], 'ml_avg_breakout_probability'] = np.mean([l['breakout_probability'] for l in nearby_levels])
                    features.loc[data.index[i], 'ml_avg_touch_probability'] = np.mean([l['touch_probability'] for l in nearby_levels])
                    features.loc[data.index[i], 'ml_level_density'] = len(nearby_levels)
                    features.loc[data.index[i], 'ml_support_resistance_ratio'] = len([l for l in nearby_levels if l['type'] == 'support']) / len(nearby_levels)
                else:
                    # Default values when no nearby levels
                    features.loc[data.index[i], 'ml_avg_reliability'] = 0.5
                    features.loc[data.index[i], 'ml_max_reliability'] = 0.5
                    features.loc[data.index[i], 'ml_avg_bounce_probability'] = 0.5
                    features.loc[data.index[i], 'ml_avg_breakout_probability'] = 0.5
                    features.loc[data.index[i], 'ml_avg_touch_probability'] = 0.5
                    features.loc[data.index[i], 'ml_level_density'] = 0.0
                    features.loc[data.index[i], 'ml_support_resistance_ratio'] = 0.5

        return features

    def _extract_trading_features(self, data: pd.DataFrame,
                                sr_levels: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Extract trading-relevant features."""
        features = pd.DataFrame(index=data.index)

        if sr_levels is None:
            return features

        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])

        # Calculate trading features for each price point
        for i, price in enumerate(data['close']):
            if pd.notna(price):
                # Find nearest support and resistance
                nearest_support = None
                nearest_resistance = None
                min_support_distance = float('inf')
                min_resistance_distance = float('inf')

                for level in support_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance < min_support_distance:
                        min_support_distance = distance
                        nearest_support = level_price

                for level in resistance_levels:
                    level_price = level.get('price', level) if isinstance(level, dict) else level
                    distance = abs(price - level_price) / price
                    if distance < min_resistance_distance:
                        min_resistance_distance = distance
                        nearest_resistance = level_price

                # Calculate trading features
                if nearest_support:
                    support_reliability = self.historical_analyzer.get_level_reliability_score(nearest_support, 'support')
                    support_probabilities = self.historical_analyzer.get_level_probability_features(nearest_support, 'support')

                    features.loc[data.index[i], 'trading_support_reliability'] = support_reliability
                    features.loc[data.index[i], 'trading_support_bounce_probability'] = support_probabilities['bounce_probability']
                    features.loc[data.index[i], 'trading_support_distance'] = min_support_distance
                    features.loc[data.index[i], 'trading_support_risk_score'] = 1.0 - support_reliability
                else:
                    features.loc[data.index[i], 'trading_support_reliability'] = 0.0
                    features.loc[data.index[i], 'trading_support_bounce_probability'] = 0.0
                    features.loc[data.index[i], 'trading_support_distance'] = 1.0
                    features.loc[data.index[i], 'trading_support_risk_score'] = 1.0

                if nearest_resistance:
                    resistance_reliability = self.historical_analyzer.get_level_reliability_score(nearest_resistance, 'resistance')
                    resistance_probabilities = self.historical_analyzer.get_level_probability_features(nearest_resistance, 'resistance')

                    features.loc[data.index[i], 'trading_resistance_reliability'] = resistance_reliability
                    features.loc[data.index[i], 'trading_resistance_breakout_probability'] = resistance_probabilities['breakout_probability']
                    features.loc[data.index[i], 'trading_resistance_distance'] = min_resistance_distance
                    features.loc[data.index[i], 'trading_resistance_risk_score'] = 1.0 - resistance_reliability
                else:
                    features.loc[data.index[i], 'trading_resistance_reliability'] = 0.0
                    features.loc[data.index[i], 'trading_resistance_breakout_probability'] = 0.0
                    features.loc[data.index[i], 'trading_resistance_distance'] = 1.0
                    features.loc[data.index[i], 'trading_resistance_risk_score'] = 1.0

                # Overall trading features
                features.loc[data.index[i], 'trading_overall_risk_score'] = (
                    features.loc[data.index[i], 'trading_support_risk_score'] +
                    features.loc[data.index[i], 'trading_resistance_risk_score']
                ) / 2

                features.loc[data.index[i], 'trading_level_zone_width'] = abs(min_resistance_distance - min_support_distance) if (nearest_support and nearest_resistance) else 1.0

        return features

def get_enhanced_sr_feature_extractor(sr_config: Optional[SRFeatureConfig] = None,
                                    historical_config: Optional[HistoricalSRConfig] = None) -> EnhancedSRFeatureExtractor:
    """Get an enhanced SR feature extractor instance."""
    return EnhancedSRFeatureExtractor(sr_config, historical_config)

def extract_enhanced_sr_features(data: pd.DataFrame,
                               sr_levels: Optional[Dict[str, Any]] = None,
                               regime_labels: Optional[pd.Series] = None,
                               sr_config: Optional[SRFeatureConfig] = None,
                               historical_config: Optional[HistoricalSRConfig] = None) -> pd.DataFrame:
    """
    Quick function to extract enhanced SR features with historical integration.

    Args:
        data: Market data with OHLCV columns
        sr_levels: Pre-computed SR levels (optional)
        regime_labels: Regime labels for regime-aware features (optional)
        sr_config: SR feature configuration (optional)
        historical_config: Historical analysis configuration (optional)

    Returns:
        DataFrame with enhanced SR features including historical analysis
    """
    extractor = get_enhanced_sr_feature_extractor(sr_config, historical_config)
    return extractor.extract_historical_sr_features(data, sr_levels, regime_labels)
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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
