"""
Enhanced Entry Quality Scoring for Tactician

This module provides advanced entry quality scoring algorithms that improve upon
the simple linear weighted formula. It includes:

1. Adaptive Multi-Factor Scoring (regime-aware with additional features)
2. Information Ratio Scoring (financial theory-based)
3. Expected Utility Scoring (risk-aversion aware)
4. ML-Based Prediction (learns from historical performance)

Default recommendation: Adaptive Multi-Factor Scoring
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from src.utils.logger import system_logger
    from src.utils.tprint import tprint_info, tprint_warning, tprint_error
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    system_logger = None


class ScoringMethod(str, Enum):
    """Available entry quality scoring methods."""
    LINEAR_WEIGHTED = "linear_weighted"  # Original simple formula
    ADAPTIVE_MULTI_FACTOR = "adaptive_multi_factor"  # Recommended
    INFORMATION_RATIO = "information_ratio"
    EXPECTED_UTILITY = "expected_utility"
    ML_BASED = "ml_based"  # Requires training


@dataclass
class EnhancedScoringConfig:
    """Configuration for enhanced entry quality scoring."""
    
    # Scoring method selection
    scoring_method: ScoringMethod = ScoringMethod.ADAPTIVE_MULTI_FACTOR
    
    # Risk parameters
    max_adverse_movement_pct: float = 0.5
    min_favorable_movement_pct: float = 0.2
    
    # Scoring thresholds
    min_quality_threshold: float = 0.25
    high_quality_threshold: float = 0.70
    
    # Risk aversion (for expected utility)
    risk_aversion: float = 2.0  # 2.0 = moderate risk aversion
    
    # Benchmark return (for information ratio)
    benchmark_return: float = 0.0
    
    # Regime-adaptive weights
    use_regime_adaptation: bool = True
    
    # Feature weights (for linear/adaptive methods)
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        'risk_reward': 0.25,
        'timing': 0.20,
        'volatility': 0.15,
        'volume': 0.15,
        'momentum': 0.15,
        'microstructure': 0.05,
        'price_action': 0.05
    })
    
    # Interaction bonuses
    enable_interaction_terms: bool = True
    interaction_bonus_cap: float = 0.20
    
    # Penalty system
    enable_penalty_system: bool = True
    max_penalty: float = 0.20


class EnhancedEntryQualityScorer:
    """
    Enhanced entry quality scoring with multiple algorithms.
    
    Provides sophisticated entry quality calculation that:
    - Adapts to market regimes
    - Includes multiple relevant factors (volume, momentum, microstructure)
    - Captures non-linear interactions
    - Applies financial theory (Information Ratio, Expected Utility)
    - Optionally learns from historical data (ML-based)
    """
    
    def __init__(self, config: Optional[EnhancedScoringConfig] = None):
        """Initialize the enhanced entry quality scorer."""
        self.config = config or EnhancedScoringConfig()
        self.logger = system_logger.getChild('EnhancedEntryQualityScorer') if system_logger else None
        
        # ML model (if ML-based scoring is used)
        self.ml_model = None
        self.ml_scaler = None
        
        if UTILS_AVAILABLE:
            tprint_info(f"✅ Enhanced Entry Quality Scorer initialized (method: {self.config.scoring_method.value})")
    
    def calculate_entry_quality(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        regime: Optional[str] = None,
        market_context: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate entry quality score using configured method.
        
        Args:
            entry_point: Current candle/entry point (Series with OHLCV)
            future_data: Future candles for forward-looking analysis
            regime: Market regime identifier (for adaptive methods)
            market_context: Additional market context (volatility, trend, liquidity)
        
        Returns:
            Quality score in range [0.0, 1.0]
        """
        if future_data.empty:
            return 0.0
        
        if market_context is None:
            market_context = {}
        
        # Select scoring method
        method = self.config.scoring_method
        
        try:
            if method == ScoringMethod.LINEAR_WEIGHTED:
                return self._calculate_linear_weighted(entry_point, future_data)
            
            elif method == ScoringMethod.ADAPTIVE_MULTI_FACTOR:
                return self._calculate_adaptive_multi_factor(
                    entry_point, future_data, regime, market_context
                )
            
            elif method == ScoringMethod.INFORMATION_RATIO:
                return self._calculate_information_ratio(entry_point, future_data)
            
            elif method == ScoringMethod.EXPECTED_UTILITY:
                return self._calculate_expected_utility(entry_point, future_data)
            
            elif method == ScoringMethod.ML_BASED:
                if self.ml_model is None:
                    warnings.warn("ML model not trained, falling back to adaptive multi-factor")
                    return self._calculate_adaptive_multi_factor(
                        entry_point, future_data, regime, market_context
                    )
                return self._calculate_ml_based(entry_point, future_data, market_context)
            
            else:
                if self.logger:
                    self.logger.warning(f"Unknown scoring method: {method}, using adaptive")
                return self._calculate_adaptive_multi_factor(
                    entry_point, future_data, regime, market_context
                )
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error calculating entry quality: {e}")
            return 0.0
    
    # ==================== SCORING METHODS ====================
    
    def _calculate_linear_weighted(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Original simple linear weighted formula.
        quality = risk_reward × 0.4 + timing × 0.3 + volatility × 0.3
        """
        risk_reward = self._calculate_risk_reward_score(entry_point, future_data)
        timing = self._calculate_timing_score(entry_point, future_data)
        volatility = self._calculate_volatility_score(entry_point, future_data)
        
        quality = risk_reward * 0.4 + timing * 0.3 + volatility * 0.3
        return float(np.clip(quality, 0.0, 1.0))
    
    def _calculate_adaptive_multi_factor(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        regime: Optional[str],
        market_context: Dict[str, float]
    ) -> float:
        """
        Adaptive multi-factor scoring with regime-aware weights.
        Recommended default method.
        """
        # 1. Calculate all component scores (normalized 0-1)
        risk_reward = self._calculate_risk_reward_score(entry_point, future_data)
        timing = self._calculate_timing_score(entry_point, future_data)
        volatility = self._calculate_volatility_score(entry_point, future_data)
        volume_quality = self._calculate_volume_quality(entry_point, future_data)
        momentum = self._calculate_momentum_alignment(entry_point, future_data)
        microstructure = self._calculate_microstructure_quality(entry_point, future_data)
        price_action = self._calculate_price_action_strength(entry_point, future_data)
        
        # 2. Get regime-adaptive weights
        if self.config.use_regime_adaptation and regime:
            weights = self._get_regime_weights(regime, market_context)
        else:
            weights = self.config.default_weights
        
        # 3. Calculate base weighted score
        base_score = (
            weights.get('risk_reward', 0.25) * risk_reward +
            weights.get('timing', 0.20) * timing +
            weights.get('volatility', 0.15) * volatility +
            weights.get('volume', 0.15) * volume_quality +
            weights.get('momentum', 0.15) * momentum +
            weights.get('microstructure', 0.05) * microstructure +
            weights.get('price_action', 0.05) * price_action
        )
        
        # 4. Apply interaction bonuses (capture synergies)
        interaction_bonus = 0.0
        if self.config.enable_interaction_terms:
            # Risk-reward + momentum synergy (both high = strong trend entry)
            if risk_reward > 0.7 and momentum > 0.7:
                interaction_bonus += 0.10 * min(risk_reward, momentum)
            
            # Timing + volume synergy (high volume at good timing)
            if timing > 0.6 and volume_quality > 0.6:
                interaction_bonus += 0.08 * (timing * volume_quality)
            
            # Volatility + microstructure synergy (stable conditions)
            if volatility > 0.5 and microstructure > 0.6:
                interaction_bonus += 0.05 * (volatility * microstructure)
            
            # Cap interaction bonus
            interaction_bonus = min(interaction_bonus, self.config.interaction_bonus_cap)
        
        # 5. Apply penalties for adverse conditions
        penalty = 0.0
        if self.config.enable_penalty_system:
            # High volatility + poor timing = risky entry
            if volatility < 0.3 and timing < 0.4:
                penalty += 0.15
            
            # Poor risk-reward despite good timing = false signal
            if risk_reward < 0.3 and timing > 0.7:
                penalty += 0.10
            
            # Low volume + high momentum = unsustainable move
            if volume_quality < 0.3 and momentum > 0.8:
                penalty += 0.08
            
            # Cap penalty
            penalty = min(penalty, self.config.max_penalty)
        
        # 6. Final score
        final_score = base_score + interaction_bonus - penalty
        return float(np.clip(final_score, 0.0, 1.0))
    
    def _calculate_information_ratio(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Information Ratio scoring: (Expected Return - Benchmark) / Tracking Error
        Based on financial theory for risk-adjusted returns.
        """
        if future_data.empty or len(future_data) < 2:
            return 0.0
        
        entry_price = entry_point['close']
        
        # Calculate return path
        future_returns = future_data['close'].pct_change().dropna()
        
        if len(future_returns) == 0:
            return 0.0
        
        # Expected return (use max favorable move as proxy)
        max_favorable = (future_data['high'].max() - entry_price) / entry_price
        
        # Tracking error (volatility of returns)
        tracking_error = future_returns.std()
        
        if tracking_error < 1e-8:
            return 0.5  # Neutral score if no volatility
        
        # Information ratio
        information_ratio = (max_favorable - self.config.benchmark_return) / tracking_error
        
        # Normalize to [0, 1] using sigmoid
        # IR=0 → score=0.5, IR=2 → score≈0.98, IR=-2 → score≈0.02
        score = 1.0 / (1.0 + np.exp(-information_ratio * 2.0))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_expected_utility(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Expected Utility Theory scoring: U = E[Return] - (λ/2) * Var[Return]
        Incorporates risk aversion parameter.
        """
        if future_data.empty or len(future_data) < 2:
            return 0.0
        
        entry_price = entry_point['close']
        
        # Calculate return distribution
        future_prices = future_data['close']
        future_returns = (future_prices - entry_price) / entry_price
        
        # Expected return and variance
        expected_return = future_returns.mean()
        return_variance = future_returns.var()
        
        # Expected utility (CARA utility function)
        # Higher risk aversion = more penalty for variance
        expected_utility = expected_return - (self.config.risk_aversion / 2.0) * return_variance
        
        # Normalize to [0, 1] using sigmoid
        score = 1.0 / (1.0 + np.exp(-expected_utility * 10.0))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_ml_based(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        market_context: Dict[str, float]
    ) -> float:
        """
        ML-based quality prediction using trained model.
        Requires prior training with historical entry performance data.
        """
        # Extract features
        features = self._extract_ml_features(entry_point, future_data, market_context)
        
        # Scale features
        if self.ml_scaler is not None:
            features_scaled = self.ml_scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Predict
        quality = self.ml_model.predict(features_scaled)[0]
        return float(np.clip(quality, 0.0, 1.0))
    
    # ==================== COMPONENT SCORES ====================
    
    def _calculate_risk_reward_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Enhanced risk-reward score with bounded calculation.
        Uses percentile-based movements for robustness.
        """
        if future_data.empty:
            return 0.0
        
        entry_price = entry_point['close']
        
        # Use percentile-based movements (more robust than max/min)
        favorable_moves = (future_data['high'] - entry_price) / entry_price * 100
        adverse_moves = (entry_price - future_data['low']) / entry_price * 100
        
        favorable_move = np.percentile(favorable_moves, 75)  # 75th percentile
        adverse_move = np.percentile(adverse_moves, 75)
        
        # Check thresholds
        if adverse_move > self.config.max_adverse_movement_pct:
            return 0.0
        
        if favorable_move < self.config.min_favorable_movement_pct:
            return 0.0
        
        # Bounded risk-reward ratio (minimum adverse movement)
        adverse_move = max(adverse_move, 0.01)
        risk_reward = favorable_move / adverse_move
        
        # Normalize using sigmoid
        # RR=3 → score≈0.95, RR=1 → score≈0.5, RR=0.5 → score≈0.27
        score = 1.0 / (1.0 + np.exp(-2.0 * (risk_reward - 1.0)))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_timing_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Timing score: earlier entries within period are better.
        """
        if future_data.empty:
            return 0.0
        
        # Time to peak (in number of candles)
        time_to_peak = len(future_data)
        
        # Normalize: shorter time = higher score
        # 1 candle → 1.0, 10 candles → 0.5, 30 candles → 0.25
        score = 1.0 / (1.0 + 0.1 * time_to_peak)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_volatility_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Volatility score: lower volatility = more stable entry = higher score.
        """
        if future_data.empty or len(future_data) < 2:
            return 0.5
        
        # Calculate realized volatility
        returns = future_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.5
        
        volatility = returns.std()
        
        # Normalize: low volatility = high score
        # vol=0.5% → 0.95, vol=2% → 0.5, vol=10% → 0.08
        score = np.exp(-volatility * 20)
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_volume_quality(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Volume quality: moderate volume with increasing trend is optimal.
        """
        if future_data.empty:
            return 0.5
        
        entry_volume = entry_point['volume']
        avg_volume = future_data['volume'].mean()
        
        if avg_volume == 0:
            return 0.5
        
        # Volume ratio (prefer moderate, not extreme)
        volume_ratio = entry_volume / avg_volume
        volume_score = 1.0 / (1.0 + np.exp(-2.0 * (volume_ratio - 1.0)))
        
        # Volume trend (increasing = better confirmation)
        if len(future_data) >= 2:
            volume_trend = np.polyfit(range(len(future_data)), future_data['volume'].values, 1)[0]
            volume_trend_normalized = np.tanh(volume_trend / avg_volume * 10)
            trend_score = (volume_trend_normalized + 1.0) / 2.0
        else:
            trend_score = 0.5
        
        # Combined score
        return float(np.clip(0.6 * volume_score + 0.4 * trend_score, 0.0, 1.0))
    
    def _calculate_momentum_alignment(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Momentum alignment: entry aligned with momentum direction.
        """
        if len(future_data) < 3:
            return 0.5
        
        # Combine current and future prices
        all_prices = pd.concat([
            pd.Series([entry_point['close']]),
            future_data['close']
        ]).reset_index(drop=True)
        
        # Calculate EMAs
        ema_short = all_prices.ewm(span=min(5, len(all_prices)), adjust=False).mean()
        ema_long = all_prices.ewm(span=min(20, len(all_prices)), adjust=False).mean()
        
        # Momentum signal (percentage)
        if len(ema_long) > 0 and ema_long.iloc[-1] != 0:
            momentum = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] * 100
        else:
            momentum = 0.0
        
        # Normalize using tanh: [-1, 1] → [0, 1]
        momentum_normalized = np.tanh(momentum / 2.0)
        score = (momentum_normalized + 1.0) / 2.0
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _calculate_microstructure_quality(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Market microstructure quality: tight spreads, low gaps, consistent volume.
        """
        if future_data.empty:
            return 0.5
        
        # 1. Effective spread (HL range)
        hl_spreads = (future_data['high'] - future_data['low']) / future_data['close']
        avg_spread = hl_spreads.mean()
        spread_score = np.exp(-avg_spread * 20)
        
        # 2. Price continuity (no large gaps)
        if len(future_data) >= 2:
            price_gaps = future_data['open'].diff().abs() / future_data['close']
            gap_score = np.exp(-price_gaps.mean() * 50)
        else:
            gap_score = 1.0
        
        # 3. Volume consistency
        if future_data['volume'].mean() > 0:
            volume_cv = future_data['volume'].std() / future_data['volume'].mean()
            volume_consistency = np.exp(-volume_cv)
        else:
            volume_consistency = 0.5
        
        # Combined microstructure score
        return float(np.clip(
            0.4 * spread_score + 0.3 * gap_score + 0.3 * volume_consistency,
            0.0, 1.0
        ))
    
    def _calculate_price_action_strength(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame
    ) -> float:
        """
        Price action strength: strong candle patterns = higher score.
        """
        # Current candle characteristics
        body_size = abs(entry_point['close'] - entry_point['open']) / entry_point['close']
        hl_range = (entry_point['high'] - entry_point['low']) / entry_point['close']
        
        # Prefer: large body (strong move), moderate range
        body_score = min(body_size * 10, 1.0)  # Normalize
        range_score = 1.0 / (1.0 + hl_range * 20)  # Prefer tight range
        
        # Combined score
        return float(np.clip(0.6 * body_score + 0.4 * range_score, 0.0, 1.0))
    
    # ==================== REGIME ADAPTATION ====================
    
    def _get_regime_weights(
        self,
        regime: str,
        market_context: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get regime-adaptive weights for scoring components.
        """
        # Start with default weights
        weights = self.config.default_weights.copy()
        
        # Adjust based on regime
        regime_lower = regime.lower() if regime else ""
        
        if 'high_vol' in regime_lower or 'volatile' in regime_lower:
            # High volatility: prioritize risk management
            weights['risk_reward'] = 0.35
            weights['volatility'] = 0.25
            weights['timing'] = 0.15
            weights['microstructure'] = 0.10
            weights['volume'] = 0.08
            weights['momentum'] = 0.05
            weights['price_action'] = 0.02
            
        elif 'trend' in regime_lower or 'directional' in regime_lower:
            # Trending: momentum and timing critical
            weights['momentum'] = 0.30
            weights['timing'] = 0.25
            weights['risk_reward'] = 0.20
            weights['price_action'] = 0.10
            weights['volume'] = 0.08
            weights['volatility'] = 0.05
            weights['microstructure'] = 0.02
            
        elif 'rang' in regime_lower or 'consolidat' in regime_lower:
            # Ranging: risk-reward and price action matter
            weights['risk_reward'] = 0.35
            weights['price_action'] = 0.20
            weights['volatility'] = 0.20
            weights['microstructure'] = 0.10
            weights['timing'] = 0.08
            weights['volume'] = 0.05
            weights['momentum'] = 0.02
            
        elif 'low_liquid' in regime_lower or 'illiquid' in regime_lower:
            # Low liquidity: volume and microstructure key
            weights['volume'] = 0.30
            weights['microstructure'] = 0.25
            weights['risk_reward'] = 0.25
            weights['volatility'] = 0.10
            weights['timing'] = 0.05
            weights['momentum'] = 0.03
            weights['price_action'] = 0.02
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    # ==================== ML SUPPORT ====================
    
    def _extract_ml_features(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        market_context: Dict[str, float]
    ) -> np.ndarray:
        """Extract features for ML model prediction."""
        features = []
        
        # Component scores
        features.append(self._calculate_risk_reward_score(entry_point, future_data))
        features.append(self._calculate_timing_score(entry_point, future_data))
        features.append(self._calculate_volatility_score(entry_point, future_data))
        features.append(self._calculate_volume_quality(entry_point, future_data))
        features.append(self._calculate_momentum_alignment(entry_point, future_data))
        features.append(self._calculate_microstructure_quality(entry_point, future_data))
        features.append(self._calculate_price_action_strength(entry_point, future_data))
        
        # Market context
        features.append(market_context.get('regime_volatility', 0.0))
        features.append(market_context.get('trend_strength', 0.0))
        features.append(market_context.get('liquidity_score', 0.0))
        
        return np.array(features)
    
    def train_ml_model(
        self,
        historical_entries: pd.DataFrame,
        actual_outcomes: pd.Series
    ):
        """
        Train ML model on historical entry performance.
        
        Args:
            historical_entries: DataFrame with entry features
            actual_outcomes: Series with actual quality scores (0-1)
                           based on realized PnL, drawdown, time to target
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            # Scale features
            self.ml_scaler = StandardScaler()
            X = self.ml_scaler.fit_transform(historical_entries)
            y = actual_outcomes.values
            
            # Train model
            self.ml_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.ml_model, X, y, cv=5, scoring='neg_mean_squared_error'
            )
            rmse = np.sqrt(-cv_scores.mean())
            
            if UTILS_AVAILABLE:
                tprint_info(f"✅ ML model trained - CV RMSE: {rmse:.4f}")
            
            # Train on full data
            self.ml_model.fit(X, y)
            
        except ImportError as e:
            if self.logger:
                self.logger.error(f"Failed to train ML model (sklearn required): {e}")
            raise


# ==================== CONVENIENCE FUNCTIONS ====================

def create_enhanced_scorer(
    method: ScoringMethod = ScoringMethod.ADAPTIVE_MULTI_FACTOR,
    **kwargs
) -> EnhancedEntryQualityScorer:
    """
    Create an enhanced entry quality scorer with specified method.
    
    Args:
        method: Scoring method to use
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured EnhancedEntryQualityScorer instance
    """
    config = EnhancedScoringConfig(scoring_method=method, **kwargs)
    return EnhancedEntryQualityScorer(config)


def compare_scoring_methods(
    entry_point: pd.Series,
    future_data: pd.DataFrame,
    regime: Optional[str] = None,
    market_context: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compare all available scoring methods for a given entry.
    
    Returns:
        Dictionary mapping method name to quality score
    """
    if market_context is None:
        market_context = {}
    
    results = {}
    
    for method in ScoringMethod:
        if method == ScoringMethod.ML_BASED:
            continue  # Skip ML (requires training)
        
        try:
            scorer = create_enhanced_scorer(method)
            score = scorer.calculate_entry_quality(
                entry_point, future_data, regime, market_context
            )
            results[method.value] = score
        except Exception as e:
            results[method.value] = None
            if UTILS_AVAILABLE:
                tprint_warning(f"Failed to calculate {method.value}: {e}")
    
    return results