"""
Enhanced SR Quality Data Collector - Stores Raw Metrics

MODIFICATION: Store BOTH heuristic scores AND raw metrics
This allows training data-driven models that learn thresholds.

Changes from original sr_quality_data_collector.py:
1. Store raw bounce percentages (not normalized)
2. Store raw hold bars (not normalized to 20)
3. Store raw volume ratios (not normalized to 2.5x)
4. Store rejection bar index (not converted to speed score)
5. Store actual trade PnL (not just normalized profit)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from datetime import timedelta

logger = logging.getLogger(__name__)


class EnhancedSRQualityDataCollector:
    """
    Enhanced data collector that stores BOTH:
    - Heuristic scores (for backward compatibility)
    - Raw metrics (for data-driven models)
    """
    
    def _measure_level_performance_enhanced(self, level, future_data: pd.DataFrame,
                                           historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        ENHANCED version that stores raw metrics.
        
        Returns both:
        - Original heuristic scores (bounce_strength, hold_strength, etc.)
        - Raw metrics (bounce_pct_raw, hold_bars_raw, etc.)
        
        This allows comparison of heuristic vs data-driven approaches.
        """
        
        # Get level info
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        if level_price is None or level_type not in ['support', 'resistance']:
            return self._get_default_performance_enhanced()
        
        tolerance = level_price * 0.005
        timeframe = getattr(self, 'current_timeframe', '1h')
        
        # Check for hits
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        else:
            hits = future_data[future_data['high'] >= level_price - tolerance]
        
        if len(hits) == 0:
            return self._get_untested_performance_enhanced()
        
        first_hit_idx = hits.index[0]
        hit_bar = hits.loc[first_hit_idx]
        early_future = future_data.loc[first_hit_idx:].iloc[:5]
        
        # ================================================================
        # 1. BOUNCE METRICS - Store BOTH heuristic AND raw
        # ================================================================
        
        weighted_bounce_pct_raw, max_bounce_pct_raw = self._calculate_time_weighted_bounce(
            early_future, hit_bar, level_type, level_price
        )
        
        # HEURISTIC: Normalize by adaptive threshold
        bounce_threshold = self._get_adaptive_bounce_threshold(timeframe)
        bounce_strength = min(weighted_bounce_pct_raw / bounce_threshold, 1.0)  # Heuristic
        max_bounce_strength = min(max_bounce_pct_raw / bounce_threshold, 1.0)
        
        # RAW: Actual percentages (let model learn threshold)
        bounce_pct_raw = weighted_bounce_pct_raw  # NEW: Store raw value
        max_bounce_pct_raw = max_bounce_pct_raw   # NEW: Store raw value
        
        # ================================================================
        # 2. HOLD METRICS - Store BOTH heuristic AND raw
        # ================================================================
        
        # Calculate bars until break
        if level_type == 'support':
            breaks = future_data.loc[first_hit_idx:][
                future_data['close'] < level_price - tolerance
            ]
        else:
            breaks = future_data.loc[first_hit_idx:][
                future_data['close'] > level_price + tolerance
            ]
        
        if len(breaks) == 0:
            bars_until_break_raw = len(future_data.loc[first_hit_idx:])  # NEW: Actual bars
            hold_strength = 1.0  # HEURISTIC: Perfect hold
            never_broke_raw = 1.0  # NEW: Binary flag
        else:
            bars_until_break_raw = len(future_data.loc[first_hit_idx:breaks.index[0]])  # NEW: Actual bars
            hold_strength = min(bars_until_break_raw / 20, 1.0)  # HEURISTIC: Normalize to 20
            never_broke_raw = 0.0  # NEW: Binary flag
        
        # ================================================================
        # 3. TRADE METRICS - Store BOTH heuristic AND raw
        # ================================================================
        
        trade_result = self._simulate_trade_enhanced(
            level_type, level_price, future_data, first_hit_idx
        )
        
        # HEURISTIC: Normalized profit score
        trade_profit = trade_result['normalized_profit']  # [-1, 1]
        
        # RAW: Actual PnL percentage
        trade_pnl_pct_raw = trade_result['actual_pnl_pct']  # NEW: Real PnL %
        trade_won_raw = trade_result['won']  # NEW: Binary outcome
        trade_bars_held_raw = trade_result['bars_held']  # NEW: How long
        
        # ================================================================
        # 4. REJECTION SPEED - Store BOTH heuristic AND raw
        # ================================================================
        
        rejection_result = self._calculate_rejection_speed_enhanced(
            future_data, hit_bar, level_type, level_price, first_hit_idx
        )
        
        # HEURISTIC: Speed score [0, 1]
        rejection_speed = rejection_result['speed_score']
        
        # RAW: Actual bar index and bounce size
        rejection_bar_index_raw = rejection_result['bar_index']  # NEW: 0-5
        rejection_bounce_pct_raw = rejection_result['bounce_pct']  # NEW: Actual %
        
        # ================================================================
        # 5. VOLUME - Store BOTH heuristic AND raw
        # ================================================================
        
        volume_result = self._calculate_volume_quality_enhanced(
            future_data, historical_data, first_hit_idx
        )
        
        # HEURISTIC: Normalized volume score
        volume_quality = volume_result['volume_score']
        
        # RAW: Actual volume ratios
        test_volume_ratio_raw = volume_result['test_ratio']  # NEW: Actual ratio
        bounce_volume_ratio_raw = volume_result['bounce_ratio']  # NEW: Actual ratio
        
        # ================================================================
        # 6. COMPOSITE QUALITY SCORES
        # ================================================================
        
        # HEURISTIC: Fixed weighted combination
        quality_score_heuristic = (
            bounce_strength * 0.25 +
            hold_strength * 0.20 +
            max(trade_profit, 0) * 0.20 +
            rejection_speed * 0.20 +
            volume_quality * 0.15
        )
        
        # DATA-DRIVEN: Let model learn weights (store all raw components)
        # Model will learn optimal combination from features
        
        # ================================================================
        # RETURN: All metrics (heuristic + raw)
        # ================================================================
        
        return {
            # HEURISTIC SCORES (backward compatible)
            'hit_rate': 1.0,
            'bounce_strength': float(bounce_strength),
            'max_bounce_strength': float(max_bounce_strength),
            'hold_strength': float(hold_strength),
            'trade_profit': float(trade_profit),
            'rejection_speed': float(rejection_speed),
            'volume_quality': float(volume_quality),
            'quality_score': float(np.clip(quality_score_heuristic, 0, 1)),
            
            # Multi-outcome heuristic scores
            'bounce_quality': float(np.clip(bounce_strength * 0.6 + rejection_speed * 0.4, 0, 1)),
            'hold_quality': float(np.clip(hold_strength * 0.7 + volume_quality * 0.3, 0, 1)),
            'trade_quality': float(np.clip(max(trade_profit, 0), 0, 1)),
            'speed_quality': float(rejection_speed),
            'volume_confirmation_quality': float(volume_quality),
            
            # ============================================================
            # NEW: RAW METRICS (for data-driven models)
            # ============================================================
            
            # Bounce raw
            'bounce_pct_raw': float(bounce_pct_raw),
            'max_bounce_pct_raw': float(max_bounce_pct_raw),
            'bounce_threshold_used': float(bounce_threshold),  # For reference
            
            # Hold raw
            'bars_until_break_raw': int(bars_until_break_raw),
            'never_broke_raw': float(never_broke_raw),
            'forward_window_bars': len(future_data.loc[first_hit_idx:]),  # Total bars observed
            
            # Trade raw
            'trade_pnl_pct_raw': float(trade_pnl_pct_raw),
            'trade_won_raw': float(trade_won_raw),
            'trade_bars_held_raw': int(trade_bars_held_raw),
            
            # Rejection raw
            'rejection_bar_index_raw': int(rejection_bar_index_raw) if rejection_bar_index_raw >= 0 else -1,
            'rejection_bounce_pct_raw': float(rejection_bounce_pct_raw),
            
            # Volume raw
            'test_volume_ratio_raw': float(test_volume_ratio_raw),
            'bounce_volume_ratio_raw': float(bounce_volume_ratio_raw),
            
            # Metadata
            'timeframe': timeframe,
            'level_type': level_type
        }
    
    def _simulate_trade_enhanced(self, level_type: str, entry_price: float,
                                 future_data: pd.DataFrame, hit_idx) -> Dict:
        """
        Simulate trade with BOTH heuristic score AND raw metrics.
        
        Returns:
            Dictionary with:
            - normalized_profit: Heuristic [-1, 1]
            - actual_pnl_pct: Raw P&L percentage
            - won: Binary outcome
            - bars_held: How many bars position was held
        """
        
        if level_type == 'support':
            stop_loss = entry_price * 0.99
            take_profit = entry_price * 1.01
            direction = 1
        else:
            stop_loss = entry_price * 1.01
            take_profit = entry_price * 0.99
            direction = -1
        
        future_bars = future_data.loc[hit_idx:].iloc[:10]
        
        for bar_idx, (_, bar) in enumerate(future_bars.iterrows()):
            if direction == 1:  # Long
                if bar['low'] <= stop_loss:
                    return {
                        'normalized_profit': -0.5,
                        'actual_pnl_pct': -0.01,  # -1% loss
                        'won': 0.0,
                        'bars_held': bar_idx + 1
                    }
                elif bar['high'] >= take_profit:
                    return {
                        'normalized_profit': 1.0,
                        'actual_pnl_pct': 0.01,  # +1% win
                        'won': 1.0,
                        'bars_held': bar_idx + 1
                    }
            else:  # Short
                if bar['high'] >= stop_loss:
                    return {
                        'normalized_profit': -0.5,
                        'actual_pnl_pct': -0.01,
                        'won': 0.0,
                        'bars_held': bar_idx + 1
                    }
                elif bar['low'] <= take_profit:
                    return {
                        'normalized_profit': 1.0,
                        'actual_pnl_pct': 0.01,
                        'won': 1.0,
                        'bars_held': bar_idx + 1
                    }
        
        # Exit at close
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return {
            'normalized_profit': np.clip(pnl_pct * 100, -1, 1),
            'actual_pnl_pct': pnl_pct,
            'won': 1.0 if pnl_pct > 0 else 0.0,
            'bars_held': len(future_bars)
        }
    
    def _calculate_rejection_speed_enhanced(self, future_data: pd.DataFrame, 
                                           hit_bar, level_type: str, 
                                           level_price: float, first_hit_idx) -> Dict:
        """
        Calculate rejection speed with BOTH heuristic AND raw metrics.
        
        Returns:
            Dictionary with:
            - speed_score: Heuristic [0, 1]
            - bar_index: Raw bar index (0-5) where rejection occurred
            - bounce_pct: Raw bounce percentage at rejection
        """
        
        early_future = future_data.loc[first_hit_idx:].iloc[:5]
        
        for i, (idx, bar) in enumerate(early_future.iterrows()):
            if 'close' not in bar or pd.isna(bar['close']):
                continue
            
            close_price = float(bar['close'])
            
            if level_type == 'support':
                bounce_size = (close_price - level_price) / level_price
            else:
                bounce_size = (level_price - close_price) / level_price
            
            if bounce_size > 0.01:  # 1% threshold
                # HEURISTIC: Position-weighted speed score
                speed_score = (1.0 - (i / 5.0)) * min(abs(bounce_size) / 0.02, 1.0)
                
                return {
                    'speed_score': float(np.clip(speed_score, 0, 1)),
                    'bar_index': i,
                    'bounce_pct': abs(bounce_size)
                }
        
        # No rejection
        return {
            'speed_score': 0.0,
            'bar_index': -1,
            'bounce_pct': 0.0
        }
    
    def _calculate_volume_quality_enhanced(self, future_data: pd.DataFrame,
                                          historical_data: pd.DataFrame,
                                          first_hit_idx) -> Dict:
        """
        Calculate volume quality with BOTH heuristic AND raw metrics.
        
        Returns:
            Dictionary with:
            - volume_score: Heuristic [0, 1]
            - test_ratio: Raw volume ratio at test
            - bounce_ratio: Raw volume ratio during bounce
        """
        
        if 'volume' not in future_data.columns or 'volume' not in historical_data.columns:
            return {
                'volume_score': 0.5,
                'test_ratio': 1.0,
                'bounce_ratio': 1.0
            }
        
        avg_volume = historical_data['volume'].mean()
        
        if pd.isna(avg_volume) or avg_volume <= 0:
            return {
                'volume_score': 0.5,
                'test_ratio': 1.0,
                'bounce_ratio': 1.0
            }
        
        # Test volume
        test_volume = future_data.loc[first_hit_idx, 'volume']
        test_volume_ratio = float(test_volume) / avg_volume
        
        # Bounce volume
        bounce_bars = future_data.loc[first_hit_idx:].iloc[:5]
        bounce_volume_avg = bounce_bars['volume'].mean()
        bounce_volume_ratio = float(bounce_volume_avg) / avg_volume if not pd.isna(bounce_volume_avg) else test_volume_ratio
        
        # HEURISTIC: Normalized by 2.5x threshold
        volume_score = (test_volume_ratio * 0.6 + bounce_volume_ratio * 0.4) / 2.5
        volume_score = float(np.clip(volume_score, 0, 1))
        
        return {
            'volume_score': volume_score,
            'test_ratio': test_volume_ratio,
            'bounce_ratio': bounce_volume_ratio
        }
    
    def _get_untested_performance_enhanced(self) -> Dict:
        """Default for untested levels - includes raw metrics."""
        return {
            # Heuristic
            'hit_rate': 0.0,
            'bounce_strength': 0.0,
            'max_bounce_strength': 0.0,
            'hold_strength': 0.5,
            'trade_profit': 0.0,
            'rejection_speed': 0.0,
            'volume_quality': 0.5,
            'quality_score': 0.2,
            'bounce_quality': 0.0,
            'hold_quality': 0.5,
            'trade_quality': 0.0,
            'speed_quality': 0.0,
            'volume_confirmation_quality': 0.5,
            
            # Raw
            'bounce_pct_raw': 0.0,
            'max_bounce_pct_raw': 0.0,
            'bounce_threshold_used': 0.04,
            'bars_until_break_raw': 0,
            'never_broke_raw': 0.0,
            'forward_window_bars': 0,
            'trade_pnl_pct_raw': 0.0,
            'trade_won_raw': 0.0,
            'trade_bars_held_raw': 0,
            'rejection_bar_index_raw': -1,
            'rejection_bounce_pct_raw': 0.0,
            'test_volume_ratio_raw': 1.0,
            'bounce_volume_ratio_raw': 1.0,
            'timeframe': '1h',
            'level_type': 'support'
        }


def compare_heuristic_vs_datadriven():
    """
    Example showing difference between heuristic and data-driven.
    """
    
    logger.info("="*80)
    logger.info("HEURISTIC vs DATA-DRIVEN COMPARISON")
    logger.info("="*80)
    
    # Example level performance
    example = {
        'bounce_pct_raw': 0.035,  # 3.5% bounce
        'bounce_threshold_used': 0.04,  # 4% threshold
        'bars_until_break_raw': 18,  # Held 18 bars
        'trade_won_raw': 1.0,  # Trade won
        'trade_pnl_pct_raw': 0.012,  # +1.2% profit
    }
    
    logger.info("\nExample Level Performance:")
    logger.info(f"  Bounce: {example['bounce_pct_raw']*100:.1f}%")
    logger.info(f"  Hold: {example['bars_until_break_raw']} bars")
    logger.info(f"  Trade: +{example['trade_pnl_pct_raw']*100:.1f}% (won)")
    
    logger.info("\nHEURISTIC APPROACH:")
    bounce_strength_heuristic = min(example['bounce_pct_raw'] / example['bounce_threshold_used'], 1.0)
    hold_strength_heuristic = min(example['bars_until_break_raw'] / 20, 1.0)
    trade_profit_heuristic = np.clip(example['trade_pnl_pct_raw'] * 100, -1, 1)
    
    quality_heuristic = (
        bounce_strength_heuristic * 0.25 +
        hold_strength_heuristic * 0.20 +
        trade_profit_heuristic * 0.20
    )
    
    logger.info(f"  Bounce strength: {bounce_strength_heuristic:.2f} (3.5% / 4% threshold)")
    logger.info(f"  Hold strength: {hold_strength_heuristic:.2f} (18 / 20 bars)")
    logger.info(f"  Trade profit: {trade_profit_heuristic:.2f} (1.2% * 100)")
    logger.info(f"  Quality score: {quality_heuristic:.2f} (0.25*0.875 + 0.20*0.90 + 0.20*1.0)")
    
    logger.info("\nDATA-DRIVEN APPROACH:")
    logger.info(f"  Bounce: 3.5% (raw) → Model learns if this is strong")
    logger.info(f"  Hold: 18 bars (raw) → Model learns if this is long enough")
    logger.info(f"  Trade: +1.2% (raw) → Model learns profitability")
    logger.info(f"  Quality score: Learned combination (not 0.25/0.20/0.20)")
    logger.info(f"")
    logger.info(f"  Benefits:")
    logger.info(f"    • May discover 3.5% is actually excellent (not 87.5%)")
    logger.info(f"    • May find 18 bars is already perfect for 1h timeframe")
    logger.info(f"    • Weights adapt to what actually predicts future performance")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compare_heuristic_vs_datadriven()

