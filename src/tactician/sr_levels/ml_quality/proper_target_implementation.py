"""
Proper Target Implementation for SR Quality Model

This shows how to modify the data collector to use REAL targets instead of heuristics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ProperTargetCalculator:
    """
    Calculate proper targets for data-driven SR quality models.
    
    Replaces heuristic quality_score with actual trading outcomes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_all_targets(self, level, future_data: pd.DataFrame, 
                              historical_data: pd.DataFrame) -> Dict:
        """
        Calculate ALL possible targets for comparison.
        
        Returns:
            Dictionary with:
            - realized_pnl_pct: PRIMARY target (actual trading P&L)
            - Raw component metrics: SECONDARY targets
            - Heuristic quality_score: For comparison only
        """
        
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        if level_price is None or level_type not in ['support', 'resistance']:
            return self._get_default_targets()
        
        tolerance = level_price * 0.005
        
        # Check if level was hit
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        else:
            hits = future_data[future_data['high'] >= level_price - tolerance]
        
        if len(hits) == 0:
            return self._get_untested_targets()
        
        first_hit_idx = hits.index[0]
        hit_bar = hits.loc[first_hit_idx]
        
        # =================================================================
        # PRIMARY TARGET: Actual Trading P&L
        # =================================================================
        
        trading_result = self._simulate_realistic_trade(
            level_type, level_price, future_data, first_hit_idx
        )
        
        realized_pnl_pct = trading_result['pnl_pct']  # PRIMARY TARGET ✓
        
        # =================================================================
        # RAW COMPONENT METRICS (for multi-task or diagnostic)
        # =================================================================
        
        # 1. Bounce metrics
        bounce_pct_raw, max_bounce_pct_raw = self._calculate_raw_bounce(
            future_data, first_hit_idx, hit_bar, level_type, level_price
        )
        
        # 2. Hold metrics
        hold_result = self._calculate_raw_hold(
            future_data, first_hit_idx, level_price, level_type, tolerance
        )
        
        # 3. Rejection metrics
        rejection_result = self._calculate_raw_rejection(
            future_data, first_hit_idx, level_price, level_type
        )
        
        # 4. Volume metrics
        volume_result = self._calculate_raw_volume(
            future_data, historical_data, first_hit_idx
        )
        
        # =================================================================
        # HEURISTIC TARGET (for comparison)
        # =================================================================
        
        # Old approach with fixed weights - keep for benchmarking
        quality_score_heuristic = self._calculate_heuristic_quality(
            bounce_pct_raw, hold_result, trading_result, rejection_result, volume_result
        )
        
        # =================================================================
        # RETURN ALL TARGETS
        # =================================================================
        
        return {
            # =========================================================
            # PRIMARY TARGET: Use this for training!
            # =========================================================
            'realized_pnl_pct': float(realized_pnl_pct),
            
            # Additional trading metrics
            'trade_won': float(trading_result['won']),
            'trade_bars_held': int(trading_result['bars_held']),
            'trade_hit_tp': float(trading_result['hit_tp']),
            'trade_hit_sl': float(trading_result['hit_sl']),
            
            # =========================================================
            # SECONDARY TARGETS: Raw component metrics
            # =========================================================
            'bounce_pct_raw': float(bounce_pct_raw),
            'max_bounce_pct_raw': float(max_bounce_pct_raw),
            
            'bars_until_break_raw': int(hold_result['bars_until_break']),
            'never_broke_raw': float(hold_result['never_broke']),
            'break_severity_pct': float(hold_result['break_severity']),
            
            'rejection_bar_index_raw': int(rejection_result['bar_index']),
            'rejection_bounce_pct_raw': float(rejection_result['bounce_pct']),
            
            'test_volume_ratio_raw': float(volume_result['test_ratio']),
            'bounce_volume_ratio_raw': float(volume_result['bounce_ratio']),
            
            # =========================================================
            # BENCHMARK: Heuristic approach (for comparison)
            # =========================================================
            'quality_score_heuristic': float(quality_score_heuristic),
            
            # =========================================================
            # METADATA
            # =========================================================
            'hit_rate': 1.0,
            'first_hit_bar': int((first_hit_idx - future_data.index[0]).total_seconds() / 3600) if hasattr(first_hit_idx, 'total_seconds') else 0,
            'level_price': float(level_price),
            'level_type': level_type
        }
    
    def _simulate_realistic_trade(self, level_type: str, entry_price: float,
                                  future_data: pd.DataFrame, hit_idx) -> Dict:
        """
        Simulate realistic trade with proper risk management.
        
        THIS IS THE PRIMARY TARGET!
        
        Setup:
        - Entry: When level is hit
        - Stop Loss: 1% below/above (risk management)
        - Take Profit: 2% above/below (2:1 R/R)
        - Max hold: 10 bars
        
        Returns:
            Dictionary with:
            - pnl_pct: Actual P&L percentage (PRIMARY TARGET)
            - won: Binary outcome
            - bars_held: Duration
            - hit_tp/hit_sl: What happened
        """
        
        # Risk management parameters
        sl_pct = 0.01  # 1% stop loss
        tp_pct = 0.02  # 2% take profit (2:1 R/R)
        max_bars = 10
        
        if level_type == 'support':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            direction = 1
        else:  # resistance
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
            direction = -1
        
        future_bars = future_data.loc[hit_idx:].iloc[:max_bars]
        
        for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
            if direction == 1:  # Long from support
                if bar['low'] <= stop_loss:
                    return {
                        'pnl_pct': -sl_pct,  # Lost 1%
                        'won': 0.0,
                        'bars_held': bar_idx + 1,
                        'hit_tp': 0.0,
                        'hit_sl': 1.0
                    }
                if bar['high'] >= take_profit:
                    return {
                        'pnl_pct': tp_pct,  # Made 2%
                        'won': 1.0,
                        'bars_held': bar_idx + 1,
                        'hit_tp': 1.0,
                        'hit_sl': 0.0
                    }
            else:  # Short from resistance
                if bar['high'] >= stop_loss:
                    return {
                        'pnl_pct': -sl_pct,
                        'won': 0.0,
                        'bars_held': bar_idx + 1,
                        'hit_tp': 0.0,
                        'hit_sl': 1.0
                    }
                if bar['low'] <= take_profit:
                    return {
                        'pnl_pct': tp_pct,
                        'won': 1.0,
                        'bars_held': bar_idx + 1,
                        'hit_tp': 1.0,
                        'hit_sl': 0.0
                    }
        
        # Neither hit - exit at market after max_bars
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return {
            'pnl_pct': pnl_pct,
            'won': 1.0 if pnl_pct > 0 else 0.0,
            'bars_held': len(future_bars),
            'hit_tp': 0.0,
            'hit_sl': 0.0
        }
    
    def _calculate_raw_bounce(self, future_data: pd.DataFrame, first_hit_idx,
                              hit_bar, level_type: str, level_price: float) -> Tuple[float, float]:
        """Calculate raw bounce percentages (no normalization)."""
        
        early_future = future_data.loc[first_hit_idx:].iloc[:5]
        
        if len(early_future) == 0:
            return 0.0, 0.0
        
        weighted_bounce = 0.0
        total_weight = 0.0
        max_bounce_pct = 0.0
        
        for i, (_, bar) in enumerate(early_future.iterrows()):
            if level_type == 'support':
                bounce = bar['high'] - hit_bar['low']
            else:
                bounce = hit_bar['high'] - bar['low']
            
            bounce_pct = bounce / level_price
            max_bounce_pct = max(max_bounce_pct, bounce_pct)
            
            # Time-weighted
            weight = np.exp(-i / 3)
            weighted_bounce += bounce_pct * weight
            total_weight += weight
        
        weighted_bounce_pct = weighted_bounce / total_weight if total_weight > 0 else 0.0
        
        return weighted_bounce_pct, max_bounce_pct
    
    def _calculate_raw_hold(self, future_data: pd.DataFrame, first_hit_idx,
                           level_price: float, level_type: str, tolerance: float) -> Dict:
        """Calculate raw hold metrics (no normalization)."""
        
        future_after_hit = future_data.loc[first_hit_idx:]
        
        if level_type == 'support':
            breaks = future_after_hit[future_after_hit['close'] < level_price - tolerance]
        else:
            breaks = future_after_hit[future_after_hit['close'] > level_price + tolerance]
        
        if len(breaks) == 0:
            return {
                'bars_until_break': len(future_after_hit),
                'never_broke': 1.0,
                'break_severity': 0.0
            }
        
        bars_until_break = len(future_after_hit.loc[:breaks.index[0]])
        
        # How bad was the break?
        break_bar = breaks.iloc[0]
        if level_type == 'support':
            break_severity = (level_price - break_bar['close']) / level_price
        else:
            break_severity = (break_bar['close'] - level_price) / level_price
        
        return {
            'bars_until_break': bars_until_break,
            'never_broke': 0.0,
            'break_severity': max(0, break_severity)
        }
    
    def _calculate_raw_rejection(self, future_data: pd.DataFrame, first_hit_idx,
                                 level_price: float, level_type: str) -> Dict:
        """Calculate raw rejection metrics."""
        
        early_future = future_data.loc[first_hit_idx:].iloc[:5]
        
        for i, (_, bar) in enumerate(early_future.iterrows()):
            close_price = bar['close']
            
            if level_type == 'support':
                bounce_size = (close_price - level_price) / level_price
            else:
                bounce_size = (level_price - close_price) / level_price
            
            if bounce_size > 0.01:  # 1% threshold
                return {
                    'bar_index': i,
                    'bounce_pct': abs(bounce_size)
                }
        
        return {
            'bar_index': -1,
            'bounce_pct': 0.0
        }
    
    def _calculate_raw_volume(self, future_data: pd.DataFrame,
                             historical_data: pd.DataFrame, first_hit_idx) -> Dict:
        """Calculate raw volume ratios."""
        
        if 'volume' not in future_data.columns or 'volume' not in historical_data.columns:
            return {'test_ratio': 1.0, 'bounce_ratio': 1.0}
        
        avg_volume = historical_data['volume'].mean()
        
        if pd.isna(avg_volume) or avg_volume <= 0:
            return {'test_ratio': 1.0, 'bounce_ratio': 1.0}
        
        test_volume = future_data.loc[first_hit_idx, 'volume']
        test_ratio = test_volume / avg_volume
        
        bounce_bars = future_data.loc[first_hit_idx:].iloc[:5]
        bounce_volume_avg = bounce_bars['volume'].mean()
        bounce_ratio = bounce_volume_avg / avg_volume if not pd.isna(bounce_volume_avg) else test_ratio
        
        return {
            'test_ratio': test_ratio,
            'bounce_ratio': bounce_ratio
        }
    
    def _calculate_heuristic_quality(self, bounce_pct_raw: float, hold_result: Dict,
                                    trading_result: Dict, rejection_result: Dict,
                                    volume_result: Dict) -> float:
        """
        Calculate OLD heuristic quality score for comparison.
        
        This is what we're trying to REPLACE with data-driven approaches.
        """
        
        # Normalize bounce (heuristic threshold: 4%)
        bounce_strength = min(bounce_pct_raw / 0.04, 1.0)
        
        # Normalize hold (heuristic threshold: 20 bars)
        hold_strength = min(hold_result['bars_until_break'] / 20, 1.0)
        
        # Normalize trade (already -1 to 1)
        trade_profit = np.clip(trading_result['pnl_pct'] * 50, -1, 1)  # Scale to [-1, 1]
        
        # Rejection speed (heuristic formula)
        if rejection_result['bar_index'] >= 0:
            rejection_speed = (1.0 - rejection_result['bar_index'] / 5.0) * \
                            min(rejection_result['bounce_pct'] / 0.02, 1.0)
        else:
            rejection_speed = 0.0
        
        # Volume quality (heuristic threshold: 2.5x)
        volume_quality = min((volume_result['test_ratio'] * 0.6 + 
                             volume_result['bounce_ratio'] * 0.4) / 2.5, 1.0)
        
        # HEURISTIC WEIGHTED COMBINATION
        quality_score = (
            bounce_strength * 0.25 +
            hold_strength * 0.20 +
            max(trade_profit, 0) * 0.20 +
            rejection_speed * 0.20 +
            volume_quality * 0.15
        )
        
        return np.clip(quality_score, 0, 1)
    
    def _get_untested_targets(self) -> Dict:
        """Targets for levels that were never hit."""
        return {
            'realized_pnl_pct': 0.0,
            'trade_won': 0.0,
            'trade_bars_held': 0,
            'trade_hit_tp': 0.0,
            'trade_hit_sl': 0.0,
            'bounce_pct_raw': 0.0,
            'max_bounce_pct_raw': 0.0,
            'bars_until_break_raw': 0,
            'never_broke_raw': 0.0,
            'break_severity_pct': 0.0,
            'rejection_bar_index_raw': -1,
            'rejection_bounce_pct_raw': 0.0,
            'test_volume_ratio_raw': 1.0,
            'bounce_volume_ratio_raw': 1.0,
            'quality_score_heuristic': 0.2,
            'hit_rate': 0.0
        }
    
    def _get_default_targets(self) -> Dict:
        """Default targets for invalid levels."""
        return self._get_untested_targets()


# Example usage and comparison
def demonstrate_proper_targets():
    """
    Show the difference between training on heuristic vs real targets.
    """
    
    logger.info("="*80)
    logger.info("PROPER TARGET DEMONSTRATION")
    logger.info("="*80)
    
    # Simulate example level
    example_metrics = {
        'bounce_pct_raw': 0.028,  # 2.8% bounce
        'bars_until_break': 25,   # Held 25 bars
        'realized_pnl_pct': 0.018,  # Made 1.8% profit (REAL MONEY)
        'rejection_bar_index': 1,  # Fast rejection
        'test_volume_ratio': 2.8,  # 2.8x volume
    }
    
    logger.info("\n📊 Example Level Performance:")
    logger.info(f"  Bounce: {example_metrics['bounce_pct_raw']*100:.1f}%")
    logger.info(f"  Hold: {example_metrics['bars_until_break']} bars")
    logger.info(f"  Trading P&L: {example_metrics['realized_pnl_pct']*100:.1f}%  ← REAL MONEY")
    logger.info(f"  Rejection: Bar {example_metrics['rejection_bar_index']}")
    logger.info(f"  Volume: {example_metrics['test_volume_ratio']:.1f}x")
    
    # Calculate heuristic quality
    bounce_heuristic = min(0.028 / 0.04, 1.0)  # 0.70 (seems weak!)
    hold_heuristic = min(25 / 20, 1.0)  # 1.00
    quality_heuristic = bounce_heuristic * 0.25 + hold_heuristic * 0.20  # = 0.375
    
    logger.info("\n❌ TRAINING ON HEURISTIC TARGET:")
    logger.info(f"  Target: quality_score_heuristic = {quality_heuristic:.3f}")
    logger.info(f"  Issue: 2.8% bounce seems 'weak' (70% of 4% threshold)")
    logger.info(f"  Issue: Model learns to predict arbitrary normalized values")
    logger.info(f"  Issue: Doesn't optimize for actual trading profit!")
    
    logger.info("\n✅ TRAINING ON REAL TARGET:")
    logger.info(f"  Target: realized_pnl_pct = {example_metrics['realized_pnl_pct']:.3f} (1.8%)")
    logger.info(f"  Benefit: Model learns this level MADE MONEY")
    logger.info(f"  Benefit: 2.8% bounce is actually strong (led to 1.8% profit)")
    logger.info(f"  Benefit: Direct optimization for trading performance")
    
    logger.info("\n💡 KEY INSIGHT:")
    logger.info("  Heuristic says: quality = 0.375 (mediocre)")
    logger.info("  Reality says: Made 1.8% profit (excellent!)")
    logger.info("  → Train on reality, not heuristics!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demonstrate_proper_targets()

