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
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the barrier calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("barrier_calculation", {})
        self.logger = system_logger.getChild("barrier_calculator")
        
        # Barrier configuration
        self.base_barriers = self.config.get("base_barriers", {
            "conservative": (0.002, 0.002),  # 0.2% up/down
            "moderate": (0.005, 0.005),      # 0.5% up/down
            "aggressive": (0.010, 0.010),    # 1.0% up/down
            "adaptive": None                 # Calculated dynamically
        })
        
        # Dynamic calculation settings
        self.use_volatility_scaling = self.config.get("use_volatility_scaling", True)
        self.use_regime_adaptation = self.config.get("use_regime_adaptation", True)
        self.volatility_window = self.config.get("volatility_window", 20)
        self.volatility_multiplier = self.config.get("volatility_multiplier", 2.0)
        
        # Asymmetric barrier settings
        self.allow_asymmetric = self.config.get("allow_asymmetric", True)
        self.risk_reward_ratio = self.config.get("risk_reward_ratio", 2.0)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="regime barrier calculation"
    )
    async def calculate_regime_barriers(
        self,
        regime_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        n_combinations: int = 4
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate regime-specific barriers.
        
        Args:
            regime_data: Market data for the regime
            regime_info: Regime metadata
            n_combinations: Number of barrier combinations to generate
            
        Returns:
            Dictionary of barrier configurations
        """
        self.logger.info(f"Calculating barriers for regime with {len(regime_data)} samples")
        
        barriers = {}
        
        # Add base barriers
        for name, barrier_values in self.base_barriers.items():
            if barrier_values is not None:
                barriers[name] = barrier_values
        
        # Calculate adaptive barriers if needed
        if self.base_barriers.get("adaptive") is None:
            adaptive_barriers = await self._calculate_adaptive_barriers(
                regime_data,
                regime_info
            )
            barriers["adaptive"] = adaptive_barriers
        
        # Generate additional combinations if needed
        if len(barriers) < n_combinations:
            additional_barriers = await self._generate_barrier_combinations(
                regime_data,
                n_combinations - len(barriers)
            )
            barriers.update(additional_barriers)
        
        # Apply regime-specific adjustments
        if self.use_regime_adaptation:
            barriers = await self._apply_regime_adjustments(
                barriers,
                regime_info
            )
        
        return barriers
    
    async def _calculate_adaptive_barriers(
        self,
        data: pd.DataFrame,
        regime_info: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate adaptive barriers based on market conditions.
        
        Args:
            data: Market data
            regime_info: Regime metadata
            
        Returns:
            Tuple of (upper_barrier, lower_barrier)
        """
        if 'close' not in data.columns or len(data) < self.volatility_window:
            # Return default if insufficient data
            return (0.005, 0.005)
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Calculate volatility
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
        
        if np.isnan(volatility) or volatility <= 0:
            volatility = returns.std()
        
        # Base barriers on volatility
        if self.use_volatility_scaling:
            upper_barrier = volatility * self.volatility_multiplier
            lower_barrier = volatility * self.volatility_multiplier
        else:
            upper_barrier = 0.005
            lower_barrier = 0.005
        
        # Apply asymmetric adjustments
        if self.allow_asymmetric:
            # Analyze recent trend
            recent_trend = returns.iloc[-self.volatility_window:].mean()
            
            if recent_trend > 0:
                # Bullish trend: wider profit target, tighter stop loss
                upper_barrier *= self.risk_reward_ratio
                lower_barrier /= np.sqrt(self.risk_reward_ratio)
            elif recent_trend < 0:
                # Bearish trend: tighter profit target, wider stop loss
                upper_barrier /= np.sqrt(self.risk_reward_ratio)
                lower_barrier *= self.risk_reward_ratio
        
        # Ensure minimum barriers
        upper_barrier = max(upper_barrier, 0.001)  # Min 0.1%
        lower_barrier = max(lower_barrier, 0.001)  # Min 0.1%
        
        return (upper_barrier, lower_barrier)
    
    async def _generate_barrier_combinations(
        self,
        data: pd.DataFrame,
        n_combinations: int
    ) -> Dict[str, Tuple[float, float]]:
        """Generate additional barrier combinations.
        
        Args:
            data: Market data
            n_combinations: Number of combinations to generate
            
        Returns:
            Dictionary of barrier configurations
        """
        combinations = {}
        
        # Calculate statistics for barrier generation
        if 'close' in data.columns and len(data) > 1:
            returns = data['close'].pct_change().dropna()
            
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90]
            return_percentiles = np.percentile(np.abs(returns), percentiles)
            
            # Generate combinations based on percentiles
            for i in range(min(n_combinations, len(percentiles))):
                barrier_value = return_percentiles[i]
                name = f"percentile_{percentiles[i]}"
                
                if self.allow_asymmetric and i < len(returns):
                    # Create asymmetric version
                    trend = returns.mean()
                    if trend > 0:
                        combinations[name] = (
                            barrier_value * self.risk_reward_ratio,
                            barrier_value
                        )
                    else:
                        combinations[name] = (
                            barrier_value,
                            barrier_value * self.risk_reward_ratio
                        )
                else:
                    combinations[name] = (barrier_value, barrier_value)
        
        # Fill remaining with scaled versions
        if len(combinations) < n_combinations:
            base_barrier = 0.005
            scales = np.linspace(0.5, 2.0, n_combinations - len(combinations))
            
            for i, scale in enumerate(scales):
                name = f"scaled_{i+1}"
                combinations[name] = (base_barrier * scale, base_barrier * scale)
        
        return combinations
    
    async def _apply_regime_adjustments(
        self,
        barriers: Dict[str, Tuple[float, float]],
        regime_info: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """Apply regime-specific adjustments to barriers.
        
        Args:
            barriers: Current barrier configurations
            regime_info: Regime metadata
            
        Returns:
            Adjusted barrier configurations
        """
        # Extract regime characteristics
        volatility_regime = regime_info.get("volatility_regime", "normal")
        trend_regime = regime_info.get("trend_regime", "neutral")
        
        adjusted_barriers = {}
        
        for name, (upper, lower) in barriers.items():
            # Adjust based on volatility regime
            if volatility_regime == "high":
                # Wider barriers in high volatility
                upper *= 1.5
                lower *= 1.5
            elif volatility_regime == "low":
                # Tighter barriers in low volatility
                upper *= 0.7
                lower *= 0.7
            
            # Adjust based on trend regime
            if trend_regime == "bullish":
                # Asymmetric for bullish regime
                upper *= 1.2
                lower *= 0.8
            elif trend_regime == "bearish":
                # Asymmetric for bearish regime
                upper *= 0.8
                lower *= 1.2
            
            adjusted_barriers[name] = (upper, lower)
        
        return adjusted_barriers
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="barrier statistics calculation"
    )
    async def calculate_barrier_statistics(
        self,
        data: pd.DataFrame,
        barriers: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Dict[str, float]]:
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
            stats = {
                "upper_barrier": upper,
                "lower_barrier": lower,
                "risk_reward_ratio": upper / lower if lower > 0 else 1.0,
                "expected_hit_rate_upper": 0.0,
                "expected_hit_rate_lower": 0.0,
                "expected_time_to_barrier": 0.0
            }
            
            # Calculate expected hit rates based on historical returns
            if len(returns) > 0:
                stats["expected_hit_rate_upper"] = (returns > upper).mean()
                stats["expected_hit_rate_lower"] = (returns < -lower).mean()
                
                # Estimate time to barrier using volatility
                volatility = returns.std()
                if volatility > 0:
                    # Simplified estimation
                    stats["expected_time_to_barrier"] = (
                        (upper + lower) / (2 * volatility)
                    )
            
            statistics[name] = stats
        
        return statistics
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=(0.005, 0.005),
        context="optimal barrier calculation"
    )
    async def calculate_optimal_barriers(
        self,
        data: pd.DataFrame,
        target_win_rate: float = 0.6,
        target_risk_reward: float = 1.5
    ) -> Tuple[float, float]:
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
        
        # Use distribution fitting to estimate optimal barriers
        # Fit a normal distribution to returns
        mu, sigma = stats.norm.fit(returns)
        
        # Calculate barriers based on target win rate
        # For a symmetric distribution, we want P(return > upper) = 1 - target_win_rate
        z_score = stats.norm.ppf(target_win_rate)
        
        # Base barrier on volatility
        base_barrier = sigma * abs(z_score)
        
        # Apply risk/reward ratio
        if target_risk_reward > 1:
            upper_barrier = base_barrier * target_risk_reward
            lower_barrier = base_barrier
        else:
            upper_barrier = base_barrier
            lower_barrier = base_barrier / target_risk_reward
        
        # Ensure reasonable bounds
        upper_barrier = np.clip(upper_barrier, 0.001, 0.05)  # 0.1% to 5%
        lower_barrier = np.clip(lower_barrier, 0.001, 0.05)  # 0.1% to 5%
        
        return (upper_barrier, lower_barrier)