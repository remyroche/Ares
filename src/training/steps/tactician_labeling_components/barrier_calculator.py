"""Barrier calculation component for tactician labeling."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger

class BarrierCalculator:
    """Handles dynamic barrier calculation for tactician labeling."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the barrier calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('barrier_calculation', {})
        self.logger = system_logger.getChild('barrier_calculator')
        self.base_barriers = self.config.get('base_barriers', {'conservative': (0.002, 0.002), 'moderate': (0.005, 0.005), 'aggressive': (0.01, 0.01), 'adaptive': None})
        self.use_volatility_scaling = self.config.get('use_volatility_scaling', True)
        self.use_regime_adaptation = self.config.get('use_regime_adaptation', True)
        self.volatility_window = self.config.get('volatility_window', 20)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 2.0)
        self.allow_asymmetric = self.config.get('allow_asymmetric', True)
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)

    @handles_errors(exceptions=(Exception,), default_return={}, context='regime barrier calculation')
    async def calculate_regime_barriers(self, regime_data: pd.DataFrame, regime_info: Dict[str, Any], n_combinations: int=4) -> Dict[str, Tuple[float, float]]:
        """Calculate regime-specific barriers.
        
        Args:
            regime_data: Market data for the regime
            regime_info: Regime metadata
            n_combinations: Number of barrier combinations to generate
            
        Returns:
            Dictionary of barrier configurations
        """
        self.logger.info(f'Calculating barriers for regime with {len(regime_data)} samples')
        barriers = {}
        for name, barrier_values in self.base_barriers.items():
            if barrier_values is not None:
                barriers[name] = barrier_values
        if self.base_barriers.get('adaptive') is None:
            adaptive_barriers = await self._calculate_adaptive_barriers(regime_data, regime_info)
            barriers['adaptive'] = adaptive_barriers
        if len(barriers) < n_combinations:
            additional_barriers = await self._generate_barrier_combinations(regime_data, n_combinations - len(barriers))
            barriers.update(additional_barriers)
        if self.use_regime_adaptation:
            barriers = await self._apply_regime_adjustments(barriers, regime_info)
        return barriers

    async def _calculate_adaptive_barriers(self, data: pd.DataFrame, regime_info: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate adaptive barriers based on market conditions.
        
        Args:
            data: Market data
            regime_info: Regime metadata
            
        Returns:
            Tuple of (upper_barrier, lower_barrier)
        """
        if 'close' not in data.columns or len(data) < self.volatility_window:
            return (0.005, 0.005)
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
        if np.isnan(volatility) or volatility <= 0:
            volatility = returns.std()
        if self.use_volatility_scaling:
            upper_barrier = volatility * self.volatility_multiplier
            lower_barrier = volatility * self.volatility_multiplier
        else:
            upper_barrier = 0.005
            lower_barrier = 0.005
        if self.allow_asymmetric:
            recent_trend = returns.iloc[-self.volatility_window:].mean()
            if recent_trend > 0:
                upper_barrier *= self.risk_reward_ratio
                lower_barrier /= np.sqrt(self.risk_reward_ratio)
            elif recent_trend < 0:
                upper_barrier /= np.sqrt(self.risk_reward_ratio)
                lower_barrier *= self.risk_reward_ratio
        upper_barrier = max(upper_barrier, 0.001)
        lower_barrier = max(lower_barrier, 0.001)
        return (upper_barrier, lower_barrier)

    async def _generate_barrier_combinations(self, data: pd.DataFrame, n_combinations: int) -> Dict[str, Tuple[float, float]]:
        """Generate additional barrier combinations.
        
        Args:
            data: Market data
            n_combinations: Number of combinations to generate
            
        Returns:
            Dictionary of barrier configurations
        """
        combinations = {}
        if 'close' in data.columns and len(data) > 1:
            returns = data['close'].pct_change().dropna()
            percentiles = [10, 25, 50, 75, 90]
            return_percentiles = np.percentile(np.abs(returns), percentiles)
            for i in range(min(n_combinations, len(percentiles))):
                barrier_value = return_percentiles[i]
                name = f'percentile_{percentiles[i]}'
                if self.allow_asymmetric and i < len(returns):
                    trend = returns.mean()
                    if trend > 0:
                        combinations[name] = (barrier_value * self.risk_reward_ratio, barrier_value)
                    else:
                        combinations[name] = (barrier_value, barrier_value * self.risk_reward_ratio)
                else:
                    combinations[name] = (barrier_value, barrier_value)
        if len(combinations) < n_combinations:
            base_barrier = 0.005
            scales = np.linspace(0.5, 2.0, n_combinations - len(combinations))
            for i, scale in enumerate(scales):
                name = f'scaled_{i + 1}'
                combinations[name] = (base_barrier * scale, base_barrier * scale)
        return combinations

    async def _apply_regime_adjustments(self, barriers: Dict[str, Tuple[float, float]], regime_info: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Apply regime-specific adjustments to barriers.
        
        Args:
            barriers: Current barrier configurations
            regime_info: Regime metadata
            
        Returns:
            Adjusted barrier configurations
        """
        volatility_regime = regime_info.get('volatility_regime', 'normal')
        trend_regime = regime_info.get('trend_regime', 'neutral')
        adjusted_barriers = {}
        for name, (upper, lower) in barriers.items():
            if volatility_regime == 'high':
                upper *= 1.5
                lower *= 1.5
            elif volatility_regime == 'low':
                upper *= 0.7
                lower *= 0.7
            if trend_regime == 'bullish':
                upper *= 1.2
                lower *= 0.8
            elif trend_regime == 'bearish':
                upper *= 0.8
                lower *= 1.2
            adjusted_barriers[name] = (upper, lower)
        return adjusted_barriers

    @handles_errors(exceptions=(Exception,), default_return={}, context='barrier statistics calculation')
    async def calculate_barrier_statistics(self, data: pd.DataFrame, barriers: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for barrier configurations.
        
        Args:
            data: Market data
            barriers: Barrier configurations
            
        Returns:
            Dictionary of statistics for each barrier
        """
        if 'close' not in data.columns or len(data) < 2:
            return {}
        returns = data['close'].pct_change().dropna()
        statistics = {}
        for name, (upper, lower) in barriers.items():
            stats = {'upper_barrier': upper, 'lower_barrier': lower, 'risk_reward_ratio': upper / lower if lower > 0 else 1.0, 'expected_hit_rate_upper': 0.0, 'expected_hit_rate_lower': 0.0, 'expected_time_to_barrier': 0.0}
            if len(returns) > 0:
                stats['expected_hit_rate_upper'] = (returns > upper).mean()
                stats['expected_hit_rate_lower'] = (returns < -lower).mean()
                volatility = returns.std()
                if volatility > 0:
                    stats['expected_time_to_barrier'] = (upper + lower) / (2 * volatility)
            statistics[name] = stats
        return statistics

    @handles_errors(exceptions=(Exception,), default_return=(0.005, 0.005), context='optimal barrier calculation')
    async def calculate_optimal_barriers(self, data: pd.DataFrame, target_win_rate: float=0.6, target_risk_reward: float=1.5) -> Tuple[float, float]:
        """Calculate optimal barriers based on target metrics.
        
        Args:
            data: Market data
            target_win_rate: Target win rate
            target_risk_reward: Target risk/reward ratio
            
        Returns:
            Tuple of (upper_barrier, lower_barrier)
        """
        if 'close' not in data.columns or len(data) < 100:
            return (0.005, 0.005)
        returns = data['close'].pct_change().dropna()
        mu, sigma = stats.norm.fit(returns)
        z_score = stats.norm.ppf(target_win_rate)
        base_barrier = sigma * abs(z_score)
        if target_risk_reward > 1:
            upper_barrier = base_barrier * target_risk_reward
            lower_barrier = base_barrier
        else:
            upper_barrier = base_barrier
            lower_barrier = base_barrier / target_risk_reward
        upper_barrier = np.clip(upper_barrier, 0.001, 0.05)
        lower_barrier = np.clip(lower_barrier, 0.001, 0.05)
        return (upper_barrier, lower_barrier)