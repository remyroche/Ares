"""
Short-Term Target Generator with Triple Barrier Method

This module implements a sophisticated target generation system for short-term price movements
(0.1% to 0.5%) using the triple barrier method to ensure we predict expected price direction
without adverse movement.

Key Features:
- Triple barrier method for each target percentage
- Direction-aware predictions (up/down movement)
- Timing predictions for optimal entry
- Adverse movement protection
- Multiple time horizons for validation
- Integration with existing Tactician architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced
from src.utils.math_validation import validate_finite, validate_positive, safe_divide

logger = system_logger.getChild('ShortTermTargetGenerator')


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier method."""
    # Target percentages (0.1% to 0.5%)
    target_percentages: List[float] = field(default_factory=lambda: [0.001, 0.002, 0.003, 0.004, 0.005])
    
    # Barrier configurations
    upper_barrier_multiplier: float = 1.0  # Same as target for profit take
    lower_barrier_multiplier: float = 0.5  # Half of target for stop loss
    
    # Time barriers
    max_hold_time_minutes: int = 15  # Maximum time to hold position
    min_hold_time_minutes: int = 1   # Minimum time before exit allowed
    
    # Direction detection
    enable_direction_detection: bool = True
    direction_threshold: float = 0.0005  # 0.05% minimum movement for direction
    
    # Validation settings
    enable_validation: bool = True
    validation_lookback_minutes: int = 30  # Look back 30 minutes for validation
    
    # Risk management
    max_adverse_movement: float = 0.002  # 0.2% maximum adverse movement
    min_risk_reward_ratio: float = 1.5   # Minimum risk/reward ratio


@dataclass
class ShortTermTarget:
    """Individual short-term target with triple barrier analysis."""
    target_percentage: float
    target_name: str
    
    # Triple barrier results
    upper_barrier: float
    lower_barrier: float
    time_barrier_minutes: int
    
    # Movement analysis
    direction: str  # 'up', 'down', 'neutral'
    movement_achieved: bool
    adverse_movement: bool
    
    # Timing analysis
    entry_timing_minutes: float
    exit_timing_minutes: float
    hold_duration_minutes: float
    
    # Risk metrics
    risk_reward_ratio: float
    max_adverse_movement: float
    confidence_score: float
    
    # Validation
    is_valid: bool
    validation_reason: str


@dataclass
class ShortTermTargets:
    """Collection of short-term targets with metadata."""
    timestamp: datetime
    symbol: str
    timeframe: str
    current_price: float
    
    # Individual targets
    targets: List[ShortTermTarget] = field(default_factory=list)
    
    # Summary metrics
    best_target: Optional[ShortTermTarget] = None
    overall_confidence: float = 0.0
    risk_score: float = 0.0
    
    # Metadata
    total_targets: int = 0
    valid_targets: int = 0
    generation_time: float = 0.0


class ShortTermTargetGenerator:
    """
    Generator for short-term price movement targets using triple barrier method.
    
    This class creates targets for 0.1% to 0.5% price movements with comprehensive
    risk management and direction detection.
    """
    
    def __init__(self, config: Optional[TripleBarrierConfig] = None):
        """
        Initialize the short-term target generator.
        
        Args:
            config: Triple barrier configuration
        """
        self.config = config or TripleBarrierConfig()
        self.logger = logger.getChild('ShortTermTargetGenerator')
        
        self.logger.info(f"🚀 Initializing ShortTermTargetGenerator")
        self.logger.info(f"📊 Target percentages: {[f'{p*100:.1f}%' for p in self.config.target_percentages]}")
        self.logger.info(f"⏰ Max hold time: {self.config.max_hold_time_minutes} minutes")
        self.logger.info(f"🛡️ Max adverse movement: {self.config.max_adverse_movement*100:.1f}%")
        
    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid price data for target generation'),
            KeyError: (None, 'Missing required price data columns'),
            IndexError: (None, 'Insufficient price data for analysis')
        },
        default_return=None,
        context='short-term target generation'
    )
    def generate_targets(
        self,
        price_data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> Optional[ShortTermTargets]:
        """
        Generate short-term targets using triple barrier method.
        
        Args:
            price_data: DataFrame with OHLCV data and timestamps
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            ShortTermTargets object with all generated targets
        """
        start_time = datetime.now()
        self.logger.info(f"🔄 Generating short-term targets for {symbol} ({timeframe})")
        
        try:
            # Validate input data
            if not self._validate_price_data(price_data):
                return None
            
            # Get current price (last available price)
            current_price = price_data['close'].iloc[-1]
            
            # Generate targets for each percentage
            targets = []
            for target_pct in self.config.target_percentages:
                target = self._generate_single_target(
                    price_data, target_pct, current_price, symbol, timeframe
                )
                if target:
                    targets.append(target)
            
            # Create targets collection
            targets_collection = ShortTermTargets(
                timestamp=start_time,
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                targets=targets,
                total_targets=len(targets),
                valid_targets=len([t for t in targets if t.is_valid]),
                generation_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Calculate summary metrics
            targets_collection = self._calculate_summary_metrics(targets_collection)
            
            self.logger.info(f"✅ Generated {len(targets)} targets in {targets_collection.generation_time:.3f}s")
            self.logger.info(f"📊 Valid targets: {targets_collection.valid_targets}/{targets_collection.total_targets}")
            
            return targets_collection
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate targets: {e}")
            return None
    
    def _validate_price_data(self, price_data: pd.DataFrame) -> bool:
        """Validate price data format and content."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in price_data.columns:
                self.logger.error(f"❌ Missing required column: {col}")
                return False
        
        if len(price_data) < self.config.max_hold_time_minutes + 10:
            self.logger.error(f"❌ Insufficient data: {len(price_data)} rows, need at least {self.config.max_hold_time_minutes + 10}")
            return False
        
        # Check for valid price data
        if price_data['close'].isna().any() or (price_data['close'] <= 0).any():
            self.logger.error("❌ Invalid price data (NaN or non-positive values)")
            return False
        
        return True
    
    def _generate_single_target(
        self,
        price_data: pd.DataFrame,
        target_percentage: float,
        current_price: float,
        symbol: str,
        timeframe: str
    ) -> Optional[ShortTermTarget]:
        """Generate a single target with triple barrier analysis."""
        
        target_name = f"{target_percentage*100:.1f}pct"
        self.logger.debug(f"🔄 Generating target: {target_name}")
        
        try:
            # Calculate barriers
            upper_barrier = current_price * (1 + target_percentage * self.config.upper_barrier_multiplier)
            lower_barrier = current_price * (1 - target_percentage * self.config.lower_barrier_multiplier)
            
            # Analyze price movement
            movement_analysis = self._analyze_price_movement(
                price_data, current_price, upper_barrier, lower_barrier, target_percentage
            )
            
            # Create target
            target = ShortTermTarget(
                target_percentage=target_percentage,
                target_name=target_name,
                upper_barrier=upper_barrier,
                lower_barrier=lower_barrier,
                time_barrier_minutes=self.config.max_hold_time_minutes,
                direction=movement_analysis['direction'],
                movement_achieved=movement_analysis['movement_achieved'],
                adverse_movement=movement_analysis['adverse_movement'],
                entry_timing_minutes=movement_analysis['entry_timing'],
                exit_timing_minutes=movement_analysis['exit_timing'],
                hold_duration_minutes=movement_analysis['hold_duration'],
                risk_reward_ratio=movement_analysis['risk_reward_ratio'],
                max_adverse_movement=movement_analysis['max_adverse_movement'],
                confidence_score=movement_analysis['confidence_score'],
                is_valid=movement_analysis['is_valid'],
                validation_reason=movement_analysis['validation_reason']
            )
            
            self.logger.debug(f"✅ Generated target {target_name}: {target.direction}, valid={target.is_valid}")
            return target
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate target {target_name}: {e}")
            return None
    
    def _analyze_price_movement(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        upper_barrier: float,
        lower_barrier: float,
        target_percentage: float
    ) -> Dict[str, Any]:
        """Analyze price movement using triple barrier method."""
        
        # Get future price data (simulate forward-looking analysis)
        future_data = price_data.iloc[-self.config.max_hold_time_minutes-1:-1].copy()
        
        # Initialize analysis results
        analysis = {
            'direction': 'neutral',
            'movement_achieved': False,
            'adverse_movement': False,
            'entry_timing': 0.0,
            'exit_timing': 0.0,
            'hold_duration': 0.0,
            'risk_reward_ratio': 0.0,
            'max_adverse_movement': 0.0,
            'confidence_score': 0.0,
            'is_valid': False,
            'validation_reason': 'Analysis pending'
        }
        
        try:
            # Determine direction based on price movement
            max_price = future_data['high'].max()
            min_price = future_data['low'].min()
            
            # Check if upper barrier is hit
            upper_hit = max_price >= upper_barrier
            # Check if lower barrier is hit
            lower_hit = min_price <= lower_barrier
            
            # Determine direction and movement achievement
            if upper_hit and not lower_hit:
                analysis['direction'] = 'up'
                analysis['movement_achieved'] = True
                analysis['entry_timing'] = self._find_entry_timing(future_data, current_price, 'up')
                analysis['exit_timing'] = self._find_exit_timing(future_data, upper_barrier, 'up')
                
            elif lower_hit and not upper_hit:
                analysis['direction'] = 'down'
                analysis['movement_achieved'] = True
                analysis['entry_timing'] = self._find_entry_timing(future_data, current_price, 'down')
                analysis['exit_timing'] = self._find_exit_timing(future_data, lower_barrier, 'down')
                
            elif upper_hit and lower_hit:
                # Both barriers hit - determine which came first
                upper_time = self._find_barrier_time(future_data, upper_barrier, 'up')
                lower_time = self._find_barrier_time(future_data, lower_barrier, 'down')
                
                if upper_time < lower_time:
                    analysis['direction'] = 'up'
                    analysis['movement_achieved'] = True
                    analysis['entry_timing'] = self._find_entry_timing(future_data, current_price, 'up')
                    analysis['exit_timing'] = upper_time
                else:
                    analysis['direction'] = 'down'
                    analysis['movement_achieved'] = True
                    analysis['entry_timing'] = self._find_entry_timing(future_data, current_price, 'down')
                    analysis['exit_timing'] = lower_time
                    
            else:
                # No barriers hit - check for significant movement
                price_change = (max_price - min_price) / current_price
                if price_change >= self.config.direction_threshold:
                    if max_price > current_price * (1 + self.config.direction_threshold):
                        analysis['direction'] = 'up'
                    elif min_price < current_price * (1 - self.config.direction_threshold):
                        analysis['direction'] = 'down'
                else:
                    analysis['direction'] = 'neutral'
            
            # Calculate risk metrics
            analysis['max_adverse_movement'] = self._calculate_max_adverse_movement(
                future_data, current_price, analysis['direction']
            )
            
            analysis['risk_reward_ratio'] = self._calculate_risk_reward_ratio(
                target_percentage, analysis['max_adverse_movement']
            )
            
            # Calculate confidence score
            analysis['confidence_score'] = self._calculate_confidence_score(analysis)
            
            # Validate target
            analysis['is_valid'], analysis['validation_reason'] = self._validate_target(analysis)
            
            # Calculate hold duration
            if analysis['movement_achieved']:
                analysis['hold_duration'] = analysis['exit_timing'] - analysis['entry_timing']
            else:
                analysis['hold_duration'] = self.config.max_hold_time_minutes
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in price movement analysis: {e}")
            analysis['validation_reason'] = f"Analysis error: {str(e)}"
            return analysis
    
    def _find_entry_timing(self, future_data: pd.DataFrame, current_price: float, direction: str) -> float:
        """Find optimal entry timing based on direction."""
        try:
            if direction == 'up':
                # Find when price starts moving up significantly
                threshold = current_price * 1.0001  # 0.01% threshold
                for i, (_, row) in enumerate(future_data.iterrows()):
                    if row['close'] >= threshold:
                        return float(i)
            elif direction == 'down':
                # Find when price starts moving down significantly
                threshold = current_price * 0.9999  # 0.01% threshold
                for i, (_, row) in enumerate(future_data.iterrows()):
                    if row['close'] <= threshold:
                        return float(i)
            
            return 0.0  # Immediate entry if no clear timing found
            
        except Exception as e:
            self.logger.error(f"❌ Error finding entry timing: {e}")
            return 0.0
    
    def _find_exit_timing(self, future_data: pd.DataFrame, barrier_price: float, direction: str) -> float:
        """Find exit timing when barrier is hit."""
        try:
            if direction == 'up':
                for i, (_, row) in enumerate(future_data.iterrows()):
                    if row['high'] >= barrier_price:
                        return float(i)
            elif direction == 'down':
                for i, (_, row) in enumerate(future_data.iterrows()):
                    if row['low'] <= barrier_price:
                        return float(i)
            
            return float(len(future_data))  # Exit at end if barrier not hit
            
        except Exception as e:
            self.logger.error(f"❌ Error finding exit timing: {e}")
            return float(len(future_data))
    
    def _find_barrier_time(self, future_data: pd.DataFrame, barrier_price: float, direction: str) -> float:
        """Find when a barrier is first hit."""
        return self._find_exit_timing(future_data, barrier_price, direction)
    
    def _calculate_max_adverse_movement(
        self, future_data: pd.DataFrame, current_price: float, direction: str
    ) -> float:
        """Calculate maximum adverse movement."""
        try:
            if direction == 'up':
                # For up movement, adverse movement is downward
                min_price = future_data['low'].min()
                adverse_movement = (current_price - min_price) / current_price
            elif direction == 'down':
                # For down movement, adverse movement is upward
                max_price = future_data['high'].max()
                adverse_movement = (max_price - current_price) / current_price
            else:
                # For neutral, calculate both directions
                max_price = future_data['high'].max()
                min_price = future_data['low'].min()
                adverse_movement = max(
                    (max_price - current_price) / current_price,
                    (current_price - min_price) / current_price
                )
            
            return max(0.0, adverse_movement)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating adverse movement: {e}")
            return 0.0
    
    def _calculate_risk_reward_ratio(self, target_percentage: float, max_adverse_movement: float) -> float:
        """Calculate risk/reward ratio."""
        try:
            if max_adverse_movement <= 0:
                return float('inf')  # No risk
            
            return target_percentage / max_adverse_movement
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk/reward ratio: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the target."""
        try:
            confidence = 0.0
            
            # Base confidence from movement achievement
            if analysis['movement_achieved']:
                confidence += 0.4
            
            # Direction confidence
            if analysis['direction'] != 'neutral':
                confidence += 0.2
            
            # Risk/reward ratio confidence
            risk_reward = analysis['risk_reward_ratio']
            if risk_reward >= 2.0:
                confidence += 0.2
            elif risk_reward >= 1.5:
                confidence += 0.1
            
            # Adverse movement confidence
            if analysis['max_adverse_movement'] <= self.config.max_adverse_movement:
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.0
    
    def _validate_target(self, analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if target meets criteria."""
        try:
            # Check if movement was achieved
            if not analysis['movement_achieved']:
                return False, "Target movement not achieved"
            
            # Check adverse movement limit
            if analysis['max_adverse_movement'] > self.config.max_adverse_movement:
                return False, f"Adverse movement too high: {analysis['max_adverse_movement']*100:.2f}%"
            
            # Check risk/reward ratio
            if analysis['risk_reward_ratio'] < self.config.min_risk_reward_ratio:
                return False, f"Risk/reward ratio too low: {analysis['risk_reward_ratio']:.2f}"
            
            # Check hold duration
            if analysis['hold_duration'] < self.config.min_hold_time_minutes:
                return False, f"Hold duration too short: {analysis['hold_duration']:.1f} minutes"
            
            # Check confidence score
            if analysis['confidence_score'] < 0.5:
                return False, f"Confidence score too low: {analysis['confidence_score']:.2f}"
            
            return True, "Target validation passed"
            
        except Exception as e:
            self.logger.error(f"❌ Error validating target: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _calculate_summary_metrics(self, targets_collection: ShortTermTargets) -> ShortTermTargets:
        """Calculate summary metrics for the targets collection."""
        try:
            valid_targets = [t for t in targets_collection.targets if t.is_valid]
            
            if not valid_targets:
                targets_collection.overall_confidence = 0.0
                targets_collection.risk_score = 1.0
                return targets_collection
            
            # Calculate overall confidence
            confidences = [t.confidence_score for t in valid_targets]
            targets_collection.overall_confidence = np.mean(confidences)
            
            # Calculate risk score (lower is better)
            adverse_movements = [t.max_adverse_movement for t in valid_targets]
            risk_rewards = [t.risk_reward_ratio for t in valid_targets]
            
            avg_adverse = np.mean(adverse_movements)
            avg_risk_reward = np.mean(risk_rewards)
            
            # Risk score: combination of adverse movement and risk/reward
            targets_collection.risk_score = avg_adverse / max(avg_risk_reward, 0.1)
            
            # Find best target (highest confidence with good risk/reward)
            best_target = max(valid_targets, key=lambda t: t.confidence_score * t.risk_reward_ratio)
            targets_collection.best_target = best_target
            
            return targets_collection
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating summary metrics: {e}")
            return targets_collection
    
    def get_target_summary(self, targets_collection: ShortTermTargets) -> Dict[str, Any]:
        """Get a summary of the generated targets."""
        try:
            summary = {
                'symbol': targets_collection.symbol,
                'timeframe': targets_collection.timeframe,
                'timestamp': targets_collection.timestamp.isoformat(),
                'current_price': targets_collection.current_price,
                'total_targets': targets_collection.total_targets,
                'valid_targets': targets_collection.valid_targets,
                'overall_confidence': targets_collection.overall_confidence,
                'risk_score': targets_collection.risk_score,
                'generation_time': targets_collection.generation_time,
                'targets': []
            }
            
            for target in targets_collection.targets:
                target_summary = {
                    'name': target.target_name,
                    'percentage': target.target_percentage,
                    'direction': target.direction,
                    'movement_achieved': target.movement_achieved,
                    'confidence_score': target.confidence_score,
                    'risk_reward_ratio': target.risk_reward_ratio,
                    'max_adverse_movement': target.max_adverse_movement,
                    'is_valid': target.is_valid,
                    'validation_reason': target.validation_reason
                }
                summary['targets'].append(target_summary)
            
            if targets_collection.best_target:
                summary['best_target'] = {
                    'name': targets_collection.best_target.target_name,
                    'direction': targets_collection.best_target.direction,
                    'confidence_score': targets_collection.best_target.confidence_score,
                    'risk_reward_ratio': targets_collection.best_target.risk_reward_ratio
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error creating target summary: {e}")
            return {'error': str(e)}


# Convenience functions
def create_short_term_target_generator(
    target_percentages: Optional[List[float]] = None,
    max_hold_time_minutes: int = 15,
    max_adverse_movement: float = 0.002
) -> ShortTermTargetGenerator:
    """Create a short-term target generator with custom configuration."""
    
    config = TripleBarrierConfig(
        target_percentages=target_percentages or [0.001, 0.002, 0.003, 0.004, 0.005],
        max_hold_time_minutes=max_hold_time_minutes,
        max_adverse_movement=max_adverse_movement
    )
    
    return ShortTermTargetGenerator(config)


def generate_short_term_targets(
    price_data: pd.DataFrame,
    symbol: str = "UNKNOWN",
    timeframe: str = "1m",
    config: Optional[TripleBarrierConfig] = None
) -> Optional[ShortTermTargets]:
    """Generate short-term targets for given price data."""
    
    generator = ShortTermTargetGenerator(config)
    return generator.generate_targets(price_data, symbol, timeframe)


# Example usage
if __name__ == "__main__":
    # Example of how to use the short-term target generator
    print("Short-Term Target Generator with Triple Barrier Method")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    n_samples = 100
    base_price = 100.0
    
    # Generate sample OHLCV data
    price_changes = np.random.normal(0, 0.001, n_samples)  # 0.1% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    price_data = pd.DataFrame(data)
    
    # Generate targets
    generator = create_short_term_target_generator()
    targets = generator.generate_targets(price_data, "BTCUSDT", "1m")
    
    if targets:
        summary = generator.get_target_summary(targets)
        print(f"✅ Generated {summary['valid_targets']}/{summary['total_targets']} valid targets")
        print(f"📊 Overall confidence: {summary['overall_confidence']:.3f}")
        print(f"🛡️ Risk score: {summary['risk_score']:.3f}")
        
        if summary.get('best_target'):
            best = summary['best_target']
            print(f"🎯 Best target: {best['name']} ({best['direction']}) - Confidence: {best['confidence_score']:.3f}")
        
        print("\n📋 Target Details:")
        for target in summary['targets']:
            status = "✅" if target['is_valid'] else "❌"
            print(f"{status} {target['name']}: {target['direction']} - "
                  f"Confidence: {target['confidence_score']:.3f}, "
                  f"R/R: {target['risk_reward_ratio']:.2f}, "
                  f"Adverse: {target['max_adverse_movement']*100:.2f}%")
    else:
        print("❌ Failed to generate targets")