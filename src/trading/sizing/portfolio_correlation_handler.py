"""
Portfolio Correlation Handler - Correlation-Adjusted Position Limits

Manages portfolio-level correlation to prevent correlated blow-ups:
- Rolling correlation matrix of active positions
- Adjusts max_portfolio_in_high_leverage based on correlation
- Per-trade correlation checks vs existing high-leverage positions
- Penalties for high correlation clusters

Correlation-based adjustments:
- High correlation (avg > 0.7): reduce limit by 30%
- Moderate correlation (0.4-0.7): reduce by 15%
- Low correlation (<0.4): no adjustment
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.tprint import tprint_info, tprint_warning, tprint_debug

logger = system_logger.getChild('PortfolioCorrelationHandler')


class PortfolioCorrelationHandler:
    """
    Handle correlation-adjusted portfolio limits for Kelly sizing.
    
    Features:
    - Rolling correlation matrix (30-day window by default)
    - Portfolio-level high-leverage limit adjustment
    - Per-trade correlation checks
    - High-leverage position tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correlation handler.
        
        Args:
            config: Configuration dictionary with correlation settings
        """
        self.config = config
        self.logger = logger.getChild('Handler')
        
        # Extract correlation configuration
        corr_config = config.get('correlation', {})
        self.enabled = corr_config.get('enabled', True)
        self.window_days = corr_config.get('window_days', 30)
        self.high_corr_threshold = corr_config.get('high_corr_threshold', 0.7)
        self.high_corr_penalty = corr_config.get('high_corr_penalty', 0.3)
        self.moderate_corr_threshold = corr_config.get('moderate_corr_threshold', 0.4)
        self.moderate_corr_penalty = corr_config.get('moderate_corr_penalty', 0.15)
        self.per_trade_corr_limit = corr_config.get('per_trade_corr_limit', 0.8)
        
        # Get safety limits
        safety_limits = config.get('safety_limits', {})
        self.high_leverage_threshold = safety_limits.get('high_leverage_threshold', 2.0)
        self.base_max_portfolio_high_lev = safety_limits.get('max_portfolio_in_high_leverage', 0.4)
        
        # Storage for price history (for correlation calculation)
        # price_history[symbol] = deque of (timestamp, price) tuples
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_days * 24))  # Assuming hourly data
        
        # Track current positions
        # positions[symbol] = {'size': float, 'leverage': float, 'entry_time': datetime}
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # Cached correlation matrix
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_corr_update: Optional[datetime] = None
        self._corr_update_interval = timedelta(hours=1)  # Update every hour
        
        if not self.enabled:
            tprint_warning("⚠️ Portfolio correlation handling is DISABLED")
            self.logger.warning("Correlation handling disabled")
        else:
            tprint_info("✅ Portfolio Correlation Handler initialized")
            self.logger.info(f"Window: {self.window_days} days, High corr threshold: {self.high_corr_threshold}")
    
    @handles_errors
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update price history for a symbol.
        
        Args:
            symbol: Asset symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
        """
        if not self.enabled:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        self.price_history[symbol].append((timestamp, price))
        
        # Invalidate cached correlation matrix
        self._correlation_matrix = None
    
    @handles_errors
    def update_position(
        self,
        symbol: str,
        size: float,
        leverage: float,
        entry_time: Optional[datetime] = None
    ) -> None:
        """
        Update current position tracking.
        
        Args:
            symbol: Asset symbol
            size: Position size (fraction of portfolio)
            leverage: Position leverage
            entry_time: Position entry time
        """
        if not self.enabled:
            return
        
        if entry_time is None:
            entry_time = datetime.now()
        
        if size > 0:
            self.positions[symbol] = {
                'size': size,
                'leverage': leverage,
                'entry_time': entry_time
            }
        elif symbol in self.positions:
            # Position closed
            del self.positions[symbol]
        
        self.logger.debug(f"Updated position: {symbol}, size={size:.4f}, leverage={leverage:.2f}")
    
    @handles_errors
    def calculate_correlation_matrix(self, current_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Calculate rolling correlation matrix from price history.
        
        Uses returns over the window period.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Correlation matrix as DataFrame (or None if insufficient data)
        """
        if not self.enabled:
            return None
        
        if current_time is None:
            current_time = datetime.now()
        
        # Check if we can use cached matrix
        if (self._correlation_matrix is not None and 
            self._last_corr_update is not None and
            (current_time - self._last_corr_update) < self._corr_update_interval):
            return self._correlation_matrix
        
        # Get symbols with sufficient price history
        symbols = [s for s in self.price_history if len(self.price_history[s]) > 20]
        
        if len(symbols) < 2:
            self.logger.debug("Insufficient symbols for correlation matrix")
            return None
        
        # Build returns DataFrame
        cutoff_time = current_time - timedelta(days=self.window_days)
        
        returns_dict = {}
        for symbol in symbols:
            # Filter to window
            prices = [(ts, p) for ts, p in self.price_history[symbol] if ts >= cutoff_time]
            
            if len(prices) < 20:
                continue
            
            # Calculate returns
            price_series = pd.Series([p for _, p in prices], index=[ts for ts, _ in prices])
            returns = price_series.pct_change().dropna()
            
            if len(returns) > 10:
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return None
        
        # Align time series (use inner join for common timestamps)
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Drop rows with any NaN
        
        if len(returns_df) < 10:
            return None
        
        # Calculate correlation matrix
        try:
            corr_matrix = returns_df.corr()
            
            # Cache the result
            self._correlation_matrix = corr_matrix
            self._last_corr_update = current_time
            
            return corr_matrix
        
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def get_high_leverage_positions(self) -> List[str]:
        """
        Get list of symbols with high leverage positions.
        
        Returns:
            List of symbols with leverage >= high_leverage_threshold
        """
        return [
            symbol for symbol, pos in self.positions.items()
            if pos['leverage'] >= self.high_leverage_threshold
        ]
    
    def calculate_portfolio_correlation_penalty(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[float, str]:
        """
        Calculate portfolio-level correlation penalty for high-leverage limit.
        
        Returns:
            Tuple of (penalty_factor, reason)
            Where penalty_factor is the multiplier for max_portfolio_in_high_leverage
            (1.0 = no penalty, 0.7 = 30% reduction)
        """
        if not self.enabled:
            return 1.0, "correlation_disabled"
        
        high_lev_symbols = self.get_high_leverage_positions()
        
        if len(high_lev_symbols) < 2:
            return 1.0, "insufficient_high_leverage_positions"
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(current_time)
        
        if corr_matrix is None:
            return 1.0, "insufficient_data_for_correlation"
        
        # Filter to high-leverage positions
        available_symbols = [s for s in high_lev_symbols if s in corr_matrix.index]
        
        if len(available_symbols) < 2:
            return 1.0, "high_leverage_symbols_not_in_corr_matrix"
        
        # Calculate average correlation among high-leverage positions
        subset_corr = corr_matrix.loc[available_symbols, available_symbols]
        
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(subset_corr, dtype=bool), k=1)
        correlations = subset_corr.where(mask).stack().values
        
        if len(correlations) == 0:
            return 1.0, "no_correlation_pairs"
        
        avg_corr = np.mean(np.abs(correlations))  # Use absolute correlation
        
        # Apply penalty based on average correlation
        if avg_corr > self.high_corr_threshold:
            penalty_factor = 1.0 - self.high_corr_penalty
            reason = f"high_correlation_{avg_corr:.2f}"
        elif avg_corr > self.moderate_corr_threshold:
            penalty_factor = 1.0 - self.moderate_corr_penalty
            reason = f"moderate_correlation_{avg_corr:.2f}"
        else:
            penalty_factor = 1.0
            reason = f"low_correlation_{avg_corr:.2f}"
        
        self.logger.debug(f"Portfolio correlation penalty: {penalty_factor:.2f} ({reason})")
        
        return penalty_factor, reason
    
    def get_adjusted_portfolio_limit(
        self,
        current_time: Optional[datetime] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get correlation-adjusted portfolio high-leverage limit.
        
        Returns:
            Tuple of (adjusted_limit, metadata)
        """
        if not self.enabled:
            return self.base_max_portfolio_high_lev, {'reason': 'correlation_disabled'}
        
        penalty_factor, reason = self.calculate_portfolio_correlation_penalty(current_time)
        
        adjusted_limit = self.base_max_portfolio_high_lev * penalty_factor
        
        metadata = {
            'base_limit': self.base_max_portfolio_high_lev,
            'adjusted_limit': adjusted_limit,
            'penalty_factor': penalty_factor,
            'reason': reason,
            'high_leverage_positions': len(self.get_high_leverage_positions())
        }
        
        return adjusted_limit, metadata
    
    @handles_errors
    def check_new_position_correlation(
        self,
        symbol: str,
        proposed_leverage: float,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, float, str]:
        """
        Check if new position has acceptable correlation with existing high-leverage positions.
        
        Args:
            symbol: Symbol for proposed position
            proposed_leverage: Proposed leverage level
            current_time: Current timestamp
            
        Returns:
            Tuple of (is_acceptable, max_correlation, reason)
        """
        if not self.enabled:
            return True, 0.0, "correlation_disabled"
        
        # Only check if proposed leverage is high
        if proposed_leverage < self.high_leverage_threshold:
            return True, 0.0, "proposed_leverage_not_high"
        
        # Get existing high-leverage positions
        high_lev_symbols = self.get_high_leverage_positions()
        
        if not high_lev_symbols:
            return True, 0.0, "no_existing_high_leverage_positions"
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(current_time)
        
        if corr_matrix is None:
            return True, 0.0, "insufficient_data_for_correlation"
        
        # Check if new symbol is in correlation matrix
        if symbol not in corr_matrix.index:
            return True, 0.0, "symbol_not_in_corr_matrix"
        
        # Get correlations with existing high-leverage positions
        correlations = []
        for existing_symbol in high_lev_symbols:
            if existing_symbol in corr_matrix.index:
                corr = corr_matrix.loc[symbol, existing_symbol]
                correlations.append(abs(corr))  # Use absolute value
        
        if not correlations:
            return True, 0.0, "no_correlation_data_available"
        
        max_corr = max(correlations)
        avg_corr = np.mean(correlations)
        
        # Check against per-trade limit
        if max_corr > self.per_trade_corr_limit:
            reason = f"max_correlation_too_high_{max_corr:.2f}_vs_{self.per_trade_corr_limit:.2f}"
            self.logger.warning(f"Position {symbol} rejected: {reason}")
            return False, max_corr, reason
        
        reason = f"correlation_acceptable_max_{max_corr:.2f}_avg_{avg_corr:.2f}"
        return True, max_corr, reason
    
    def calculate_correlation_adjusted_size(
        self,
        symbol: str,
        base_size: float,
        proposed_leverage: float,
        current_time: Optional[datetime] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Calculate correlation-adjusted position size.
        
        If correlation with existing high-leverage positions is too high,
        reduce the position size to conservative level.
        
        Args:
            symbol: Symbol for proposed position
            base_size: Base position size (before correlation adjustment)
            proposed_leverage: Proposed leverage
            current_time: Current timestamp
            
        Returns:
            Tuple of (adjusted_size, was_adjusted, metadata)
        """
        if not self.enabled:
            return base_size, False, {'reason': 'correlation_disabled'}
        
        is_acceptable, max_corr, reason = self.check_new_position_correlation(
            symbol, proposed_leverage, current_time
        )
        
        metadata = {
            'is_acceptable': is_acceptable,
            'max_correlation': max_corr,
            'reason': reason,
            'base_size': base_size
        }
        
        if is_acceptable:
            return base_size, False, metadata
        
        # Reduce to conservative size (50% of base for high correlation)
        adjusted_size = base_size * 0.5
        
        metadata['adjusted_size'] = adjusted_size
        metadata['reduction_pct'] = 50.0
        
        self.logger.warning(f"Reduced position size for {symbol} due to high correlation: {base_size:.4f} → {adjusted_size:.4f}")
        
        return adjusted_size, True, metadata
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """
        Get current portfolio statistics.
        
        Returns:
            Dictionary with portfolio stats
        """
        total_positions = len(self.positions)
        high_lev_positions = len(self.get_high_leverage_positions())
        
        total_exposure = sum(pos['size'] * pos['leverage'] for pos in self.positions.values())
        high_lev_exposure = sum(
            pos['size'] * pos['leverage']
            for symbol, pos in self.positions.items()
            if pos['leverage'] >= self.high_leverage_threshold
        )
        
        avg_leverage = np.mean([pos['leverage'] for pos in self.positions.values()]) if self.positions else 0.0
        
        return {
            'total_positions': total_positions,
            'high_leverage_positions': high_lev_positions,
            'total_exposure': total_exposure,
            'high_leverage_exposure': high_lev_exposure,
            'high_leverage_exposure_pct': high_lev_exposure / total_exposure if total_exposure > 0 else 0.0,
            'avg_leverage': avg_leverage,
            'symbols_tracked': len(self.price_history),
            'correlation_matrix_cached': self._correlation_matrix is not None
        }
    
    def reset(self) -> None:
        """Reset all tracking (for backtesting)."""
        self.price_history.clear()
        self.positions.clear()
        self._correlation_matrix = None
        self._last_corr_update = None
        self.logger.info("Reset correlation handler")

