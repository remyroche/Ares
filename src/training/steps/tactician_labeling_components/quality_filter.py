"""Quality filter component for tactician labeling."""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class QualityFilter:
    """Handles quality filtering for tactician labeling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the quality filter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("quality_filters", {})
        self.logger = system_logger.getChild("quality_filter")
        
        # Filter configuration
        self.min_volume_threshold = self.config.get("min_volume_threshold", 1000)
        self.min_spread_threshold = self.config.get("min_spread_threshold", 0.0001)
        self.max_spread_threshold = self.config.get("max_spread_threshold", 0.01)
        self.volatility_filter = self.config.get("volatility_filter", True)
        self.min_volatility = self.config.get("min_volatility", 0.0001)
        self.max_volatility = self.config.get("max_volatility", 0.1)
        
        # Time-based filters
        self.exclude_market_open = self.config.get("exclude_market_open", True)
        self.market_open_minutes = self.config.get("market_open_minutes", 30)
        self.exclude_market_close = self.config.get("exclude_market_close", True)
        self.market_close_minutes = self.config.get("market_close_minutes", 30)
        
        # Price-based filters
        self.min_price = self.config.get("min_price", 1.0)
        self.outlier_detection = self.config.get("outlier_detection", True)
        self.outlier_std_threshold = self.config.get("outlier_std_threshold", 3.0)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="regime filter calculation"
    )
    async def get_regime_filters(
        self,
        regime_id: str,
        regime_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get regime-specific quality filters.
        
        Args:
            regime_id: Regime identifier
            regime_data: Market data for the regime
            
        Returns:
            Dictionary of regime-specific filters
        """
        self.logger.info(f"Calculating quality filters for regime {regime_id}")
        
        filters = {
            "regime_id": regime_id,
            "base_filters": {
                "min_volume": self.min_volume_threshold,
                "min_spread": self.min_spread_threshold,
                "max_spread": self.max_spread_threshold,
                "min_price": self.min_price
            }
        }
        
        # Calculate regime-specific thresholds
        if 'volume' in regime_data.columns:
            # Adjust volume threshold based on regime average
            avg_volume = regime_data['volume'].mean()
            if not np.isnan(avg_volume):
                filters["min_volume"] = max(
                    self.min_volume_threshold * 0.5,
                    avg_volume * 0.1  # At least 10% of regime average
                )
        
        if 'spread' in regime_data.columns or ('ask' in regime_data.columns and 'bid' in regime_data.columns):
            # Calculate spread if needed
            if 'spread' not in regime_data.columns and 'ask' in regime_data.columns and 'bid' in regime_data.columns:
                spread = regime_data['ask'] - regime_data['bid']
            else:
                spread = regime_data['spread']
            
            # Adjust spread thresholds based on regime
            avg_spread = spread.mean()
            std_spread = spread.std()
            
            if not np.isnan(avg_spread) and not np.isnan(std_spread):
                filters["min_spread"] = max(
                    self.min_spread_threshold,
                    avg_spread - 2 * std_spread
                )
                filters["max_spread"] = min(
                    self.max_spread_threshold,
                    avg_spread + 2 * std_spread
                )
        
        # Calculate volatility filters
        if self.volatility_filter and 'close' in regime_data.columns:
            returns = regime_data['close'].pct_change().dropna()
            if len(returns) > 20:
                volatility = returns.rolling(window=20).std().mean()
                if not np.isnan(volatility):
                    filters["min_volatility"] = max(
                        self.min_volatility,
                        volatility * 0.2
                    )
                    filters["max_volatility"] = min(
                        self.max_volatility,
                        volatility * 5.0
                    )
        
        return filters
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="regime filter application"
    )
    async def apply_regime_filters(
        self,
        data: pd.DataFrame,
        filters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply regime-specific quality filters.
        
        Args:
            data: Market data
            filters: Filter configuration
            
        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data
        
        filtered = data.copy()
        initial_count = len(filtered)
        
        # Apply volume filter
        if 'volume' in filtered.columns and 'min_volume' in filters:
            volume_mask = filtered['volume'] >= filters['min_volume']
            filtered = filtered[volume_mask]
            self.logger.info(
                f"Volume filter: {initial_count} -> {len(filtered)} "
                f"({initial_count - len(filtered)} removed)"
            )
        
        # Apply spread filter
        if self._has_spread_data(filtered):
            spread = self._calculate_spread(filtered)
            
            if 'min_spread' in filters:
                spread_mask = spread >= filters['min_spread']
                filtered = filtered[spread_mask]
            
            if 'max_spread' in filters:
                spread_mask = spread <= filters['max_spread']
                filtered = filtered[spread_mask]
                
            self.logger.info(
                f"Spread filter: {initial_count} -> {len(filtered)} "
                f"({initial_count - len(filtered)} removed)"
            )
        
        # Apply price filter
        if 'close' in filtered.columns and 'min_price' in filters.get('base_filters', {}):
            price_mask = filtered['close'] >= filters['base_filters']['min_price']
            filtered = filtered[price_mask]
        
        # Apply volatility filter
        if self.volatility_filter and 'close' in filtered.columns and len(filtered) > 20:
            returns = filtered['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            if 'min_volatility' in filters:
                vol_mask = volatility >= filters['min_volatility']
                filtered = filtered[vol_mask]
            
            if 'max_volatility' in filters:
                vol_mask = volatility <= filters['max_volatility']
                filtered = filtered[vol_mask]
        
        # Apply time-based filters
        filtered = await self._apply_time_filters(filtered)
        
        # Apply outlier detection
        if self.outlier_detection:
            filtered = await self._remove_outliers(filtered)
        
        self.logger.info(
            f"Total filtered: {initial_count} -> {len(filtered)} "
            f"({(len(filtered)/initial_count)*100:.1f}% retained)"
        )
        
        return filtered
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="global filter application"
    )
    async def apply_global_filters(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply global quality filters across all regimes.
        
        Args:
            data: Combined labeled data
            
        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data
        
        filtered = data.copy()
        
        # Remove samples with no labels
        if 'label' in filtered.columns:
            labeled_mask = filtered['label'] != 0
            filtered = filtered[labeled_mask]
        
        # Remove samples with low signal strength
        if 'signal_strength' in filtered.columns:
            strength_mask = filtered['signal_strength'] > 0.5
            filtered = filtered[strength_mask]
        
        # Ensure balanced dataset if needed
        if 'label' in filtered.columns and self.config.get("balance_classes", False):
            filtered = await self._balance_classes(filtered)
        
        return filtered
    
    async def _apply_time_filters(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply time-based quality filters.
        
        Args:
            data: Market data with datetime index
            
        Returns:
            Filtered DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        filtered = data.copy()
        
        # Extract time components
        times = filtered.index.time
        
        # Market open filter
        if self.exclude_market_open:
            # Assuming market opens at 9:30 AM
            market_open = pd.Timestamp('09:30:00').time()
            open_cutoff = pd.Timestamp(f'09:{30 + self.market_open_minutes}:00').time()
            
            open_mask = ~((times >= market_open) & (times < open_cutoff))
            filtered = filtered[open_mask]
        
        # Market close filter
        if self.exclude_market_close:
            # Assuming market closes at 4:00 PM
            close_cutoff = pd.Timestamp(f'{16 - self.market_close_minutes//60}:{60 - self.market_close_minutes%60}:00').time()
            market_close = pd.Timestamp('16:00:00').time()
            
            close_mask = ~((times >= close_cutoff) & (times < market_close))
            filtered = filtered[close_mask]
        
        return filtered
    
    async def _remove_outliers(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Remove statistical outliers from data.
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with outliers removed
        """
        if 'close' not in data.columns or len(data) < 100:
            return data
        
        filtered = data.copy()
        
        # Calculate returns
        returns = filtered['close'].pct_change()
        
        # Remove extreme return outliers
        mean_return = returns.mean()
        std_return = returns.std()
        
        if not np.isnan(mean_return) and not np.isnan(std_return) and std_return > 0:
            lower_bound = mean_return - self.outlier_std_threshold * std_return
            upper_bound = mean_return + self.outlier_std_threshold * std_return
            
            outlier_mask = (returns >= lower_bound) & (returns <= upper_bound)
            filtered = filtered[outlier_mask | returns.isna()]
        
        # Remove price spike outliers
        if 'high' in filtered.columns and 'low' in filtered.columns:
            price_range = filtered['high'] - filtered['low']
            avg_range = price_range.rolling(window=20).mean()
            
            spike_mask = price_range <= avg_range * 5  # Remove 5x spikes
            filtered = filtered[spike_mask | avg_range.isna()]
        
        return filtered
    
    async def _balance_classes(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Balance classes in labeled data.
        
        Args:
            data: Labeled data
            
        Returns:
            Balanced DataFrame
        """
        if 'label' not in data.columns:
            return data
        
        # Get class counts
        class_counts = data['label'].value_counts()
        
        if len(class_counts) < 2:
            return data
        
        # Find minority class size
        min_class_size = class_counts.min()
        
        # Sample each class to match minority
        balanced_dfs = []
        for label in class_counts.index:
            class_data = data[data['label'] == label]
            if len(class_data) > min_class_size:
                # Downsample majority class
                sampled = class_data.sample(n=min_class_size, random_state=42)
            else:
                sampled = class_data
            balanced_dfs.append(sampled)
        
        # Combine and sort by index
        balanced = pd.concat(balanced_dfs).sort_index()
        
        return balanced
    
    def _has_spread_data(self, data: pd.DataFrame) -> bool:
        """Check if spread data is available."""
        return 'spread' in data.columns or ('ask' in data.columns and 'bid' in data.columns)
    
    def _calculate_spread(self, data: pd.DataFrame) -> pd.Series:
        """Calculate spread from bid/ask or return existing spread."""
        if 'spread' in data.columns:
            return data['spread']
        elif 'ask' in data.columns and 'bid' in data.columns:
            return data['ask'] - data['bid']
        else:
            return pd.Series(index=data.index, dtype=float)