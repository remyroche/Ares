"""
SR Level Role Reversal Detection System

This module detects when Support levels become Resistance and vice versa
after breakouts, following the classic technical analysis principle:
- Broken Support → Becomes Resistance
- Broken Resistance → Becomes Support

This is a fundamental concept in technical analysis that reflects market psychology:
when a support level is broken, traders who bought at that level are now underwater
and will often sell when price returns to that level, creating resistance.

Author: Enhanced SR Detection Team
Date: 2025-11-01
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from .enhanced_sr_detection import SRLevel

# Try to import logger, fallback to standard logging if not available
try:
    from src.utils.logger import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


class SRRoleReversalDetector:
    """
    Detects and tracks role reversal in Support/Resistance levels.
    
    Key Principles:
    1. Broken Support → Becomes Resistance
    2. Broken Resistance → Becomes Support
    3. Reversal strength increases with repeated rejections
    
    Algorithm:
    1. Identify breakouts using ATR-normalized thresholds
    2. Track post-breakout price behavior
    3. Detect tests of broken levels
    4. Confirm role reversal if rejections occur
    5. Score reversal strength based on rejection rate
    """
    
    def __init__(
        self,
        breakout_threshold: float = 1.0,  # ATR multiplier for breakout confirmation
        reversal_test_window: int = 20,   # Bars to look forward after breakout
        min_tests_for_reversal: int = 2,  # Minimum tests to confirm reversal
        rejection_threshold: float = 0.5,  # ATR multiplier for rejection detection
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize role reversal detector.
        
        Args:
            breakout_threshold: ATR multiplier for breakout confirmation (1.0 = 1 ATR)
            reversal_test_window: Number of bars to look ahead after breakout
            min_tests_for_reversal: Minimum number of tests needed to confirm reversal
            rejection_threshold: ATR multiplier for rejection detection
            logger: Optional logger instance
        """
        self.breakout_threshold = breakout_threshold
        self.reversal_test_window = reversal_test_window
        self.min_tests_for_reversal = min_tests_for_reversal
        self.rejection_threshold = rejection_threshold
        self.logger = logger or get_logger('SRRoleReversalDetector')
    
    def detect_role_reversals(
        self,
        levels: List[SRLevel],
        market_data: pd.DataFrame,
        atr: pd.Series
    ) -> List[SRLevel]:
        """
        Detect role reversals for all SR levels.
        
        Process:
        1. Identify breakouts
        2. Track post-breakout price behavior
        3. Detect tests of broken levels
        4. Confirm role reversal if rejections occur
        
        Args:
            levels: List of SR levels to analyze
            market_data: OHLCV data with DatetimeIndex
            atr: Average True Range series
            
        Returns:
            Updated list of SR levels with role reversal metadata
        """
        if not levels:
            self.logger.info("No levels to analyze for role reversals")
            return levels
        
        self.logger.info(f"🔄 Analyzing {len(levels)} SR levels for role reversals...")
        
        updated_levels = []
        reversals_detected = 0
        
        for i, level in enumerate(levels):
            # Analyze this level for role reversal
            try:
                updated_level = self._analyze_level_for_reversal(
                    level, market_data, atr
                )
                updated_levels.append(updated_level)
                
                if updated_level.role_reversed:
                    reversals_detected += 1
                
                # Progress logging every 20 levels
                if (i + 1) % 20 == 0:
                    self.logger.debug(f"   Processed {i + 1}/{len(levels)} levels...")
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze level at {level.price}: {e}")
                updated_levels.append(level)  # Keep original if analysis fails
        
        self.logger.info(f"✅ Role reversal analysis complete: {reversals_detected} reversals detected")
        
        return updated_levels
    
    def _analyze_level_for_reversal(
        self,
        level: SRLevel,
        market_data: pd.DataFrame,
        atr: pd.Series
    ) -> SRLevel:
        """
        Analyze a single level for role reversal.
        
        Args:
            level: SR level to analyze
            market_data: OHLCV data
            atr: Average True Range series
            
        Returns:
            Updated SR level with reversal information
        """
        # Initialize original_type if not set
        if level.original_type is None:
            level.original_type = level.type
        
        # Step 1: Detect if level was broken
        breakout_info = self._detect_breakout(level, market_data, atr)
        
        if not breakout_info['broken']:
            # Level not broken, no role reversal possible
            return level
        
        # Step 2: Analyze post-breakout behavior
        reversal_info = self._analyze_post_breakout_behavior(
            level, market_data, atr, breakout_info
        )
        
        # Step 3: Update level with reversal information
        if reversal_info['reversal_confirmed']:
            level = self._apply_role_reversal(level, reversal_info)
        
        return level
    
    def _detect_breakout(
        self,
        level: SRLevel,
        market_data: pd.DataFrame,
        atr: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect if and when a level was broken.
        
        A breakout is confirmed when price closes beyond the level by at least
        breakout_threshold * ATR.
        
        Args:
            level: SR level to check
            market_data: OHLCV data
            atr: Average True Range series
            
        Returns:
            {
                'broken': bool,
                'breakout_time': pd.Timestamp or None,
                'breakout_index': int or None,
                'breakout_direction': str or None  # 'up' or 'down'
            }
        """
        level_price = level.price
        original_type = level.original_type or level.type
        
        # Start searching from formation time if available
        start_idx = 0
        if level.formation_time is not None and level.formation_time in market_data.index:
            loc = market_data.index.get_loc(level.formation_time)
            # get_loc can return int, slice, or ndarray - ensure we get an int
            if isinstance(loc, int):
                start_idx = loc
            elif isinstance(loc, slice):
                start_idx = loc.start if loc.start is not None else 0
            elif hasattr(loc, '__iter__'):
                # It's an array of boolean values or indices
                start_idx = int(loc[0]) if len(loc) > 0 else 0
            else:
                start_idx = 0
        
        for i in range(start_idx, len(market_data)):
            current_atr = atr.iloc[i] if i < len(atr) and not pd.isna(atr.iloc[i]) else atr.mean()
            threshold = current_atr * self.breakout_threshold
            
            close_price = market_data['close'].iloc[i]
            
            if original_type == 'support':
                # Support broken if close is significantly below level
                if close_price < (level_price - threshold):
                    return {
                        'broken': True,
                        'breakout_time': market_data.index[i],
                        'breakout_index': i,
                        'breakout_direction': 'down'
                    }
            
            elif original_type == 'resistance':
                # Resistance broken if close is significantly above level
                if close_price > (level_price + threshold):
                    return {
                        'broken': True,
                        'breakout_time': market_data.index[i],
                        'breakout_index': i,
                        'breakout_direction': 'up'
                    }
        
        return {
            'broken': False,
            'breakout_time': None,
            'breakout_index': None,
            'breakout_direction': None
        }
    
    def _analyze_post_breakout_behavior(
        self,
        level: SRLevel,
        market_data: pd.DataFrame,
        atr: pd.Series,
        breakout_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze price behavior after breakout to detect role reversal.
        
        Key Logic:
        - After breaking support, does price bounce off it as resistance?
        - After breaking resistance, does price bounce off it as support?
        
        A test is detected when price comes within rejection_threshold * ATR of the level.
        A rejection is confirmed when price reverses away from the level.
        
        Args:
            level: SR level being analyzed
            market_data: OHLCV data
            atr: Average True Range series
            breakout_info: Breakout detection results
            
        Returns:
            {
                'reversal_confirmed': bool,
                'reversal_score': float (0-1),
                'tests': int,
                'rejections': int,
                'breakout_time': pd.Timestamp,
                'new_type': str
            }
        """
        breakout_idx = breakout_info['breakout_index']
        level_price = level.price
        original_type = level.original_type or level.type
        
        # Look ahead after breakout
        end_idx = min(breakout_idx + self.reversal_test_window, len(market_data))
        
        if end_idx <= breakout_idx:
            return {
                'reversal_confirmed': False,
                'reversal_score': 0.0,
                'tests': 0,
                'rejections': 0,
                'breakout_time': breakout_info['breakout_time'],
                'new_type': original_type
            }
        
        post_breakout_data = market_data.iloc[breakout_idx:end_idx]
        
        tests = 0
        rejections = 0
        
        for i in range(len(post_breakout_data)):
            current_idx = breakout_idx + i
            if current_idx >= len(atr):
                break
                
            current_atr = atr.iloc[current_idx] if not pd.isna(atr.iloc[current_idx]) else atr.mean()
            tolerance = current_atr * self.rejection_threshold
            
            # Get OHLC data
            high = post_breakout_data['high'].iloc[i]
            low = post_breakout_data['low'].iloc[i]
            close = post_breakout_data['close'].iloc[i]
            
            # Check if price tested the level
            if abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance:
                tests += 1
                
                # Check for rejection (role reversal confirmation)
                if original_type == 'support':
                    # Now should act as resistance - price should reject downward
                    # Level was broken downward, now price came back up to test it
                    # If it bounces down (close < level), that's a rejection
                    if high >= (level_price - tolerance) and close < level_price:
                        rejections += 1
                
                elif original_type == 'resistance':
                    # Now should act as support - price should reject upward
                    # Level was broken upward, now price came back down to test it
                    # If it bounces up (close > level), that's a rejection
                    if low <= (level_price + tolerance) and close > level_price:
                        rejections += 1
        
        # Calculate reversal confirmation score
        reversal_score = 0.0
        reversal_confirmed = False
        
        if tests >= self.min_tests_for_reversal and rejections > 0:
            reversal_score = rejections / tests
            reversal_confirmed = True
        
        # Determine new type after reversal
        new_type = 'resistance' if original_type == 'support' else 'support'
        
        return {
            'reversal_confirmed': reversal_confirmed,
            'reversal_score': reversal_score,
            'tests': tests,
            'rejections': rejections,
            'breakout_time': breakout_info['breakout_time'],
            'new_type': new_type
        }
    
    def _apply_role_reversal(
        self,
        level: SRLevel,
        reversal_info: Dict[str, Any]
    ) -> SRLevel:
        """
        Apply role reversal to the level.
        
        Args:
            level: SR level to update
            reversal_info: Reversal detection results
            
        Returns:
            Updated SR level
        """
        # Initialize type history if not exists
        if level.type_history is None:
            level.type_history = []
        
        # Record the type change
        level.type_history.append({
            'timestamp': reversal_info['breakout_time'],
            'old_type': level.type,
            'new_type': reversal_info['new_type'],
            'reversal_score': reversal_info['reversal_score'],
            'tests': reversal_info['tests'],
            'rejections': reversal_info['rejections']
        })
        
        # Update level properties
        level.type = reversal_info['new_type']
        level.role_reversed = True
        level.role_reversal_time = reversal_info['breakout_time']
        level.role_reversal_count += 1
        level.post_breakout_tests = reversal_info['tests']
        level.post_breakout_rejections = reversal_info['rejections']
        level.reversal_confirmation_score = reversal_info['reversal_score']
        
        return level
    
    def get_reversal_statistics(self, levels: List[SRLevel]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about role reversals.
        
        Args:
            levels: List of SR levels
            
        Returns:
            Dictionary with reversal statistics
        """
        if not levels:
            return {
                'total_levels': 0,
                'reversed_levels': 0,
                'reversal_rate': 0.0,
                'support_to_resistance': 0,
                'resistance_to_support': 0,
                'avg_reversal_score': 0.0,
                'avg_post_breakout_tests': 0.0,
                'avg_post_breakout_rejections': 0.0,
                'max_reversal_count': 0
            }
        
        reversed_levels = [l for l in levels if l.role_reversed]
        
        support_to_resistance = [
            l for l in reversed_levels 
            if l.original_type == 'support' and l.type == 'resistance'
        ]
        
        resistance_to_support = [
            l for l in reversed_levels 
            if l.original_type == 'resistance' and l.type == 'support'
        ]
        
        return {
            'total_levels': len(levels),
            'reversed_levels': len(reversed_levels),
            'reversal_rate': len(reversed_levels) / len(levels) if levels else 0.0,
            'support_to_resistance': len(support_to_resistance),
            'resistance_to_support': len(resistance_to_support),
            'avg_reversal_score': float(np.mean([l.reversal_confirmation_score for l in reversed_levels])) if reversed_levels else 0.0,
            'avg_post_breakout_tests': float(np.mean([l.post_breakout_tests for l in reversed_levels])) if reversed_levels else 0.0,
            'avg_post_breakout_rejections': float(np.mean([l.post_breakout_rejections for l in reversed_levels])) if reversed_levels else 0.0,
            'max_reversal_count': max([l.role_reversal_count for l in levels], default=0)
        }

