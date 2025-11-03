from ..core.decorators import handles_errors
import warnings
"""
from ..utils.logger import system_logger
Enhanced Regime Classifier for Strategist
Implements refined market regime detection with more granular regime types
"""
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from ..utils.logger import system_logger
from typing import Tuple
import pandas as pd
from typing import Dict
from typing import Any
import numpy as np
import logging
import time

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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

class EnhancedRegimeClassifier:
    """
    Enhanced regime classifier with refined regime types for strategy generation.

    Refined Regimes:
    1. STRONG_BULL: Strong uptrend with high momentum
    2. MODERATE_BULL: Steady uptrend with normal momentum
    3. WEAK_BULL: Weak uptrend, potential reversal
    4. STRONG_BEAR: Strong downtrend with high momentum
    5. MODERATE_BEAR: Steady downtrend with normal momentum
    6. WEAK_BEAR: Weak downtrend, potential reversal
    7. RANGING_HIGH: Sideways movement in upper price range
    8. RANGING_LOW: Sideways movement in lower price range
    9. VOLATILE_BULLISH: High volatility with bullish bias
    10. VOLATILE_BEARISH: High volatility with bearish bias
    11. BREAKOUT_UP: Breaking resistance levels
    12. BREAKOUT_DOWN: Breaking support levels
    """
    REGIMES = ['STRONG_BULL', 'MODERATE_BULL', 'WEAK_BULL', 'STRONG_BEAR', 'MODERATE_BEAR', 'WEAK_BEAR', 'RANGING_HIGH', 'RANGING_LOW', 'VOLATILE_BULLISH', 'VOLATILE_BEARISH', 'BREAKOUT_UP', 'BREAKOUT_DOWN']

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('EnhancedRegimeClassifier')
        self.n_states = 12
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.momentum_threshold = config.get('momentum_threshold', 0.001)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.breakout_threshold = config.get('breakout_threshold', 0.03)
        self.short_window = config.get('short_window', 5)
        self.medium_window = config.get('medium_window', 20)
        self.long_window = config.get('long_window', 50)

    @handles_errors(fallback = None)
    async def initialize(self) -> bool:
        """Initialize the enhanced regime classifier."""
        try:
            self.logger.info('Initializing Enhanced Regime Classifier...')
            # Use mode-appropriate n_iter values (defaulting to blank mode equivalent)
            n_iter = 20  # Blank mode equivalent for strategist regime classifier
            self.hmm_model = hmm.GaussianHMM(n_components = self.n_states, covariance_type='diag', n_iter = n_iter, random_state = 42)
            self.logger.info('✅ Enhanced Regime Classifier initialized')
            return True
        except Exception as e:
            self.logger.error(f'Failed to initialize regime classifier: {e}')
            return False

    def calculate_enhanced_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced features for refined regime detection.

        Features include:
        - Multi-timeframe momentum
        - Volatility patterns
        - Volume dynamics
        - Price position relative to moving averages
        - Trend strength indicators
        """
        features = pd.DataFrame(index = market_data.index)
        features['return_1'] = market_data['close'].pct_change(1)
        features['return_5'] = market_data['close'].pct_change(self.short_window)
        features['return_20'] = market_data['close'].pct_change(self.medium_window)
        features['momentum_short'] = features['return_5'].rolling(self.short_window).mean()
        features['momentum_medium'] = features['return_20'].rolling(self.medium_window).mean()
        features['momentum_acceleration'] = features['momentum_short'] - features['momentum_short'].shift(self.short_window)
        features['volatility_short'] = features['return_1'].rolling(self.short_window).std()
        features['volatility_medium'] = features['return_1'].rolling(self.medium_window).std()
        features['volatility_ratio'] = features['volatility_short'] / features['volatility_medium']
        features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(self.medium_window).mean()
        features['volume_trend'] = market_data['volume'].rolling(self.short_window).mean() / market_data['volume'].rolling(self.medium_window).mean()
        sma_short = market_data['close'].rolling(self.short_window).mean()
        sma_medium = market_data['close'].rolling(self.medium_window).mean()
        sma_long = market_data['close'].rolling(self.long_window).mean()
        features['price_position_short'] = (market_data['close'] - sma_short) / sma_short
        features['price_position_medium'] = (market_data['close'] - sma_medium) / sma_medium
        features['price_position_long'] = (market_data['close'] - sma_long) / sma_long
        features['trend_strength'] = (sma_short - sma_long) / sma_long
        features['trend_consistency'] = features['return_5'].rolling(self.medium_window).apply(lambda x: np.sum(x > 0) / len(x))
        high_rolling = market_data['high'].rolling(self.medium_window).max()
        low_rolling = market_data['low'].rolling(self.medium_window).min()
        features['distance_from_high'] = (high_rolling - market_data['close']) / market_data['close']
        features['distance_from_low'] = (market_data['close'] - low_rolling) / market_data['close']
        features['range_position'] = (market_data['close'] - low_rolling) / (high_rolling - low_rolling)
        return features.fillna(0)

    def classify_regime(self, features: pd.Series, hmm_state: int = None) -> str:
        """
        Classify market regime based on enhanced features.

        Args:
            features: Current feature values
            hmm_state: Optional HMM state for additional context

        Returns:
            Refined regime classification
        """
        momentum_short = features['momentum_short']
        momentum_medium = features['momentum_medium']
        momentum_accel = features['momentum_acceleration']
        volatility_ratio = features['volatility_ratio']
        volume_ratio = features['volume_ratio']
        trend_strength = features['trend_strength']
        distance_from_high = features['distance_from_high']
        distance_from_low = features['distance_from_low']
        if distance_from_high < 0.001 and volume_ratio > self.volume_threshold:
            return 'BREAKOUT_UP'
        elif distance_from_low < 0.001 and volume_ratio > self.volume_threshold:
            return 'BREAKOUT_DOWN'
        if volatility_ratio > 1.5:
            if momentum_short > self.momentum_threshold:
                return 'VOLATILE_BULLISH'
            elif momentum_short < -self.momentum_threshold:
                return 'VOLATILE_BEARISH'
        if trend_strength > 0.02:
            if momentum_accel > 0 and volume_ratio > 1.2:
                return 'STRONG_BULL'
            elif momentum_short > self.momentum_threshold:
                return 'MODERATE_BULL'
            else:
                return 'WEAK_BULL'
        elif trend_strength < -0.02:
            if momentum_accel < 0 and volume_ratio > 1.2:
                return 'STRONG_BEAR'
            elif momentum_short < -self.momentum_threshold:
                return 'MODERATE_BEAR'
            else:
                return 'WEAK_BEAR'
        elif features['range_position'] > 0.7:
            return 'RANGING_HIGH'
        elif features['range_position'] < 0.3:
            return 'RANGING_LOW'
        elif momentum_short > 0:
            return 'WEAK_BULL'
        else:
            return 'WEAK_BEAR'

    @handles_errors(fallback=('MODERATE_BULL', 0.5, {}))
    async def predict_regime(self, market_data: pd.DataFrame) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict current market regime with confidence.

        Args:
            market_data: Recent market data

        Returns:
            Tuple of (regime, confidence, metadata)
        """
        try:
            features_df = self.calculate_enhanced_features(market_data)
            if features_df.empty or len(features_df) < self.long_window:
                return ('MODERATE_BULL', 0.3, {'error': 'Insufficient data'})
            current_features = features_df.iloc[-1]
            hmm_state = None
            hmm_confidence = 0.5
            if self.trained and self.hmm_model is not None:
                hmm_features = features_df[['return_1', 'return_5', 'return_20', 'volatility_short', 'volatility_medium', 'volume_ratio', 'trend_strength']].iloc[-self.medium_window:].values
                hmm_features_scaled = self.scaler.transform(hmm_features)
                hmm_state = self.hmm_model.predict(hmm_features_scaled)[-1]
                log_prob, posteriors = self.hmm_model.score_samples(hmm_features_scaled)
                hmm_confidence = np.max(posteriors[-1])
            regime = self.classify_regime(current_features, hmm_state)
            rule_confidence = self._calculate_regime_confidence(current_features, regime)
            final_confidence = 0.7 * rule_confidence + 0.3 * hmm_confidence if self.trained else rule_confidence
            metadata = {'hmm_state': int(hmm_state) if hmm_state is not None else None, 'hmm_confidence': float(hmm_confidence), 'rule_confidence': float(rule_confidence), 'momentum_short': float(current_features['momentum_short']), 'momentum_medium': float(current_features['momentum_medium']), 'volatility_ratio': float(current_features['volatility_ratio']), 'trend_strength': float(current_features['trend_strength']), 'volume_ratio': float(current_features['volume_ratio']), 'timestamp': datetime.now().isoformat()}
            return (regime, final_confidence, metadata)
        except Exception as e:
            self.logger.error(f'Error predicting regime: {e}')
            return ('MODERATE_BULL', 0.3, {'error': str(e)})

    def _calculate_regime_confidence(self, features: pd.Series, regime: str) -> float:
        """Calculate confidence for regime classification based on feature strength."""
        confidence = 0.5
        momentum_strength = abs(features['momentum_short'])
        if momentum_strength > self.momentum_threshold * 2:
            confidence += 0.2
        elif momentum_strength > self.momentum_threshold:
            confidence += 0.1
        if features['trend_consistency'] > 0.7:
            confidence += 0.15
        elif features['trend_consistency'] > 0.6:
            confidence += 0.1
        if features['volume_ratio'] > 1.5:
            confidence += 0.1
        if 'BREAKOUT' in regime:
            if features['volume_ratio'] > 2.0:
                confidence += 0.2
            else:
                confidence += 0.1
        return min(confidence, 0.95)

    async def train(self, historical_data: pd.DataFrame) -> bool:
        """Train HMM model on historical data."""
        try:
            self.logger.info('Training Enhanced Regime Classifier...')
            features_df = self.calculate_enhanced_features(historical_data)
            hmm_features = features_df[['return_1', 'return_5', 'return_20', 'volatility_short', 'volatility_medium', 'volume_ratio', 'trend_strength']].dropna().values
            hmm_features_scaled = self.scaler.fit_transform(hmm_features)
            self.hmm_model.fit(hmm_features_scaled)
            self._validate_and_fix_transition_matrix()
            self.trained = True
            self.logger.info('✅ Enhanced Regime Classifier trained successfully')
            return True
        except Exception as e:
            self.logger.error(f'Failed to train regime classifier: {e}')
            return False

    def _validate_and_fix_transition_matrix(self) -> None:
        """
        Validate and fix transition matrix after training to prevent zero-sum rows.
        """
        try:
            if not hasattr(self.hmm_model, 'transmat_') or self.hmm_model.transmat_ is None:
                return

            n_components = self.hmm_model.transmat_.shape[0]

            # Check if transition matrix has any zero-sum rows
            row_sums = self.hmm_model.transmat_.sum(axis=1)
            zero_sum_rows = np.where(np.abs(row_sums) < 1e-10)[0]

            if len(zero_sum_rows) > 0:
                self.logger.warning(f"⚠️ Found {len(zero_sum_rows)} zero-sum rows in transition matrix: {zero_sum_rows}")

                # Fix zero-sum rows by setting them to uniform distribution
                for row_idx in zero_sum_rows:
                    # Set uniform transition probabilities with slight bias towards self-transition
                    uniform_prob = (1.0 - 0.7) / (n_components - 1) if n_components > 1 else 1.0
                    self.hmm_model.transmat_[row_idx, :] = uniform_prob
                    if n_components > 1:
                        self.hmm_model.transmat_[row_idx, row_idx] = 0.7  # Higher self-transition probability

                # Renormalize all rows to ensure they sum to 1
                self.hmm_model.transmat_ = self.hmm_model.transmat_ / self.hmm_model.transmat_.sum(axis=1, keepdims=True)

                self.logger.info(f"✅ Fixed {len(zero_sum_rows)} zero-sum rows in transition matrix")

            # Additional validation: ensure no NaN or infinite values
            if np.any(np.isnan(self.hmm_model.transmat_)) or np.any(np.isinf(self.hmm_model.transmat_)):
                self.logger.warning("⚠️ Found NaN or infinite values in transition matrix, applying regularization")

                # Replace NaN/inf with regularized uniform distribution
                epsilon = 1e-6
                regularized_transmat = np.full((n_components, n_components), epsilon)
                np.fill_diagonal(regularized_transmat, 0.7)

                # Distribute remaining probability
                for i in range(n_components):
                    remaining_prob = 1.0 - regularized_transmat[i, i]
                    other_states_prob = remaining_prob / (n_components - 1) if n_components > 1 else 0.0
                    for j in range(n_components):
                        if i != j:
                            regularized_transmat[i, j] = other_states_prob

                # Normalize and assign
                regularized_transmat = regularized_transmat / regularized_transmat.sum(axis=1, keepdims=True)
                self.hmm_model.transmat_ = regularized_transmat.astype(np.float64)

                self.logger.info("✅ Applied regularization to fix NaN/infinite values in transition matrix")

            # Final validation
            final_row_sums = self.hmm_model.transmat_.sum(axis=1)
            if not np.allclose(final_row_sums, 1.0, atol=1e-6):
                self.logger.warning(f"⚠️ Transition matrix rows do not sum to 1: {final_row_sums}")
                # Force normalization
                self.hmm_model.transmat_ = self.hmm_model.transmat_ / self.hmm_model.transmat_.sum(axis=1, keepdims=True)
                self.logger.info("✅ Forced normalization of transition matrix")

        except Exception as e:
            self.logger.error(f"❌ Error validating transition matrix: {e}")
            # Fallback: create a safe uniform transition matrix
            if hasattr(self.hmm_model, 'transmat_') and self.hmm_model.transmat_ is not None:
                n_components = self.hmm_model.transmat_.shape[0]
                safe_transmat = np.full((n_components, n_components), 1.0 / n_components)
                self.hmm_model.transmat_ = safe_transmat.astype(np.float64)
                self.logger.info("✅ Applied fallback uniform transition matrix")

    def get_regime_strategy_params(self, regime: str) -> Dict[str, Any]:
        """
        Get strategy parameters based on regime.

        Returns regime-specific parameters for:
        - Position sizing
        - Risk management
        - Entry/exit thresholds
        - Indicator weights
        """
        regime_params = {'STRONG_BULL': {'position_size_multiplier': 1.5, 'stop_loss_multiplier': 0.8, 'take_profit_multiplier': 1.5, 'entry_confidence_threshold': 0.6, 'momentum_weight': 0.7, 'mean_reversion_weight': 0.3}, 'MODERATE_BULL': {'position_size_multiplier': 1.0, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 1.2, 'entry_confidence_threshold': 0.65, 'momentum_weight': 0.6, 'mean_reversion_weight': 0.4}, 'WEAK_BULL': {'position_size_multiplier': 0.7, 'stop_loss_multiplier': 1.2, 'take_profit_multiplier': 1.0, 'entry_confidence_threshold': 0.7, 'momentum_weight': 0.4, 'mean_reversion_weight': 0.6}, 'STRONG_BEAR': {'position_size_multiplier': 1.5, 'stop_loss_multiplier': 0.8, 'take_profit_multiplier': 1.5, 'entry_confidence_threshold': 0.6, 'momentum_weight': 0.7, 'mean_reversion_weight': 0.3}, 'MODERATE_BEAR': {'position_size_multiplier': 1.0, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 1.2, 'entry_confidence_threshold': 0.65, 'momentum_weight': 0.6, 'mean_reversion_weight': 0.4}, 'WEAK_BEAR': {'position_size_multiplier': 0.7, 'stop_loss_multiplier': 1.2, 'take_profit_multiplier': 1.0, 'entry_confidence_threshold': 0.7, 'momentum_weight': 0.4, 'mean_reversion_weight': 0.6}, 'RANGING_HIGH': {'position_size_multiplier': 0.8, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 0.8, 'entry_confidence_threshold': 0.75, 'momentum_weight': 0.3, 'mean_reversion_weight': 0.7}, 'RANGING_LOW': {'position_size_multiplier': 0.8, 'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 0.8, 'entry_confidence_threshold': 0.75, 'momentum_weight': 0.3, 'mean_reversion_weight': 0.7}, 'VOLATILE_BULLISH': {'position_size_multiplier': 0.6, 'stop_loss_multiplier': 1.5, 'take_profit_multiplier': 1.8, 'entry_confidence_threshold': 0.8, 'momentum_weight': 0.5, 'mean_reversion_weight': 0.5}, 'VOLATILE_BEARISH': {'position_size_multiplier': 0.6, 'stop_loss_multiplier': 1.5, 'take_profit_multiplier': 1.8, 'entry_confidence_threshold': 0.8, 'momentum_weight': 0.5, 'mean_reversion_weight': 0.5}, 'BREAKOUT_UP': {'position_size_multiplier': 1.2, 'stop_loss_multiplier': 0.7, 'take_profit_multiplier': 2.0, 'entry_confidence_threshold': 0.65, 'momentum_weight': 0.8, 'mean_reversion_weight': 0.2}, 'BREAKOUT_DOWN': {'position_size_multiplier': 1.2, 'stop_loss_multiplier': 0.7, 'take_profit_multiplier': 2.0, 'entry_confidence_threshold': 0.65, 'momentum_weight': 0.8, 'mean_reversion_weight': 0.2}}
        return regime_params.get(regime, regime_params['MODERATE_BULL'])

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
