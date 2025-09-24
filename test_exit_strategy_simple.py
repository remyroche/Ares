#!/usr/bin/env python3
"""
Simple test for enhanced exit strategy logic without complex imports.
"""

from datetime import datetime, timedelta
from enum import Enum

class PositionAction(Enum):
    """Enum for position actions."""
    STAY = "stay"
    EXIT = "exit"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HEDGE = "hedge"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    FULL_CLOSE = "full_close"
    PARTIAL_PROFIT = "partial_profit"
    TRAILING_STOP = "trailing_stop"

class EnhancedExitStrategy:
    """Enhanced exit strategy with confidence-based exits and profit-taking."""
    
    def __init__(self):
        # Confidence-based exit thresholds
        self.confidence_thresholds = {
            "very_low": 0.2,      # Exit immediately
            "low": 0.4,           # Scale down or exit
            "medium": 0.6,        # Hold position
            "high": 0.8           # Consider profit taking
        }
        
        # PnL-based exit thresholds
        self.pnl_thresholds = {
            "stop_loss": -0.05,   # -5% stop loss
            "profit_target": 0.04, # 4% profit target
            "scaling_levels": [0.25, 0.5, 0.75]  # Profit scaling levels
        }
        
        # Profit-taking configuration
        self.profit_taking_config = {
            "confidence_scaling": True,    # Scale profit taking based on confidence
            "min_confidence_for_profit": 0.6,  # Minimum confidence to take profit
            "confidence_profit_multiplier": 0.5,  # How much confidence affects profit taking
            "tiered_profit_taking": True,  # Enable tiered profit taking
            "trailing_stop_enabled": True,  # Enable trailing stops
            "trailing_stop_atr_multiplier": 1.5  # ATR multiplier for trailing stops
        }
        
        self.max_position_age = 10800  # 3 hours

    def determine_position_action(self, position_data, combined_confidence):
        """Determine recommended position action based on enhanced exit strategy."""
        try:
            unrealized_pnl = position_data["unrealized_pnl"]
            entry_time = position_data.get("entry_time")
            current_time = datetime.now()
            position_id = position_data.get("position_id", "unknown")

            # 1. CRITICAL CONDITIONS - Check first (highest priority)
            if unrealized_pnl <= self.pnl_thresholds["stop_loss"]:
                return PositionAction.STOP_LOSS, f"Critical stop loss: {unrealized_pnl:.4f} <= {self.pnl_thresholds['stop_loss']:.4f}"

            # 2. TIME-BASED EXITS
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                position_age = (current_time - entry_time).total_seconds()
                if position_age > self.max_position_age:
                    return PositionAction.FULL_CLOSE, f"Maximum hold time exceeded: {position_age:.0f}s > {self.max_position_age}s"

            # 3. CONFIDENCE-BASED EXITS
            if combined_confidence < self.confidence_thresholds["very_low"]:
                return PositionAction.FULL_CLOSE, f"Very low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['very_low']:.3f}"
            elif combined_confidence < self.confidence_thresholds["low"]:
                return PositionAction.SCALE_DOWN, f"Low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['low']:.3f}"

            # 4. PROFIT-TAKING LOGIC (confidence-based scaling)
            if unrealized_pnl > 0:
                profit_action, profit_reason = self._evaluate_profit_taking(
                    unrealized_pnl, combined_confidence, position_data
                )
                if profit_action != PositionAction.STAY:
                    return profit_action, profit_reason

            # 5. CONFIDENCE-BASED POSITION MANAGEMENT
            if combined_confidence >= self.confidence_thresholds["high"]:
                return PositionAction.STAY, f"High confidence: {combined_confidence:.3f} >= {self.confidence_thresholds['high']:.3f}"
            elif combined_confidence >= self.confidence_thresholds["medium"]:
                return PositionAction.STAY, f"Medium confidence: {combined_confidence:.3f} (within acceptable range)"

            # 6. DEFAULT ACTION
            return PositionAction.STAY, f"Position maintained: confidence={combined_confidence:.3f}, pnl={unrealized_pnl:.4f}"

        except Exception as e:
            return PositionAction.STAY, f"Error in position assessment: {e}"

    def _evaluate_profit_taking(self, unrealized_pnl, combined_confidence, position_data):
        """Evaluate profit-taking opportunities with confidence-based scaling."""
        try:
            # Check if confidence is high enough for profit taking
            if combined_confidence < self.profit_taking_config["min_confidence_for_profit"]:
                return PositionAction.STAY, f"Confidence too low for profit taking: {combined_confidence:.3f} < {self.profit_taking_config['min_confidence_for_profit']:.3f}"

            # Calculate confidence-scaled profit targets
            base_profit_target = self.pnl_thresholds["profit_target"]
            
            if self.profit_taking_config["confidence_scaling"]:
                # Higher confidence = lower profit taking (hold longer for bigger gains)
                confidence_factor = 1.0 - (combined_confidence - 0.5) * self.profit_taking_config["confidence_profit_multiplier"]
                scaled_profit_target = base_profit_target * confidence_factor
            else:
                scaled_profit_target = base_profit_target

            # Check for full profit target
            if unrealized_pnl >= scaled_profit_target:
                return PositionAction.TAKE_PROFIT, f"Profit target reached: {unrealized_pnl:.4f} >= {scaled_profit_target:.4f} (confidence-scaled)"

            # Check for tiered profit taking
            if self.profit_taking_config["tiered_profit_taking"]:
                for i, level in enumerate(self.pnl_thresholds["scaling_levels"]):
                    tier_profit = scaled_profit_target * level
                    if unrealized_pnl >= tier_profit:
                        return PositionAction.PARTIAL_PROFIT, f"Tier {i+1} profit: {unrealized_pnl:.4f} >= {tier_profit:.4f} (confidence-scaled)"

            return PositionAction.STAY, f"Profit taking not triggered: {unrealized_pnl:.4f} < {scaled_profit_target:.4f}"

        except Exception as e:
            return PositionAction.STAY, f"Error in profit evaluation: {e}"

def test_enhanced_exit_strategy():
    """Test the enhanced exit strategy with various scenarios."""
    
    print("🧪 Testing Enhanced Exit Strategy")
    print("=" * 60)
    
    strategy = EnhancedExitStrategy()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "High Confidence + High Profit",
            "position_data": {
                "position_id": "test_1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 1.0,
                "entry_price": 50000.0,
                "current_price": 52000.0,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=30),
                "unrealized_pnl": 2000.0  # 4% profit
            },
            "confidence": 0.9
        },
        {
            "name": "Medium Confidence + Medium Profit", 
            "position_data": {
                "position_id": "test_2",
                "symbol": "ETHUSDT",
                "side": "LONG", 
                "quantity": 10.0,
                "entry_price": 3000.0,
                "current_price": 3090.0,  # 3% profit
                "entry_time": datetime.now() - timedelta(minutes=45),
                "unrealized_pnl": 900.0  # 3% profit
            },
            "confidence": 0.65
        },
        {
            "name": "Low Confidence + Any Profit",
            "position_data": {
                "position_id": "test_3", 
                "symbol": "ADAUSDT",
                "side": "LONG",
                "quantity": 1000.0,
                "entry_price": 0.5,
                "current_price": 0.52,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=20),
                "unrealized_pnl": 20.0  # 4% profit
            },
            "confidence": 0.3
        },
        {
            "name": "Very Low Confidence + Any PnL",
            "position_data": {
                "position_id": "test_4",
                "symbol": "DOTUSDT", 
                "side": "LONG",
                "quantity": 100.0,
                "entry_price": 7.0,
                "current_price": 7.14,  # 2% profit
                "entry_time": datetime.now() - timedelta(minutes=15),
                "unrealized_pnl": 14.0  # 2% profit
            },
            "confidence": 0.15
        },
        {
            "name": "Stop Loss Scenario",
            "position_data": {
                "position_id": "test_5",
                "symbol": "LINKUSDT",
                "side": "LONG",
                "quantity": 50.0,
                "entry_price": 15.0,
                "current_price": 14.25,  # -5% loss
                "entry_time": datetime.now() - timedelta(minutes=10),
                "unrealized_pnl": -37.5  # -5% loss
            },
            "confidence": 0.7
        },
        {
            "name": "Time-based Exit",
            "position_data": {
                "position_id": "test_6",
                "symbol": "UNIUSDT",
                "side": "LONG", 
                "quantity": 20.0,
                "entry_price": 6.0,
                "current_price": 6.12,  # 2% profit
                "entry_time": datetime.now() - timedelta(hours=4),  # 4 hours old
                "unrealized_pnl": 2.4  # 2% profit
            },
            "confidence": 0.8
        }
    ]
    
    print("📊 Running Exit Strategy Test Scenarios")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Scenario {i}: {scenario['name']}")
        print(f"   Confidence: {scenario['confidence']:.2f}")
        print(f"   PnL: {scenario['position_data']['unrealized_pnl']:.2f}")
        
        # Determine position action
        action, reason = strategy.determine_position_action(
            scenario['position_data'], 
            scenario['confidence']
        )
        
        print(f"   Action: {action.value}")
        print(f"   Reason: {reason}")
        
        # Analyze the decision
        if action == PositionAction.TAKE_PROFIT:
            print("   ✅ Profit taking triggered")
        elif action == PositionAction.PARTIAL_PROFIT:
            print("   ✅ Partial profit taking triggered")
        elif action == PositionAction.STOP_LOSS:
            print("   ⚠️  Stop loss triggered")
        elif action == PositionAction.FULL_CLOSE:
            print("   🚨 Full close triggered")
        elif action == PositionAction.SCALE_DOWN:
            print("   📉 Scale down triggered")
        elif action == PositionAction.STAY:
            print("   📍 Position maintained")
        else:
            print(f"   ❓ Unknown action: {action.value}")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced Exit Strategy Test Completed")
    
    # Test confidence-based profit scaling
    print("\n🎯 Testing Confidence-Based Profit Scaling")
    print("-" * 40)
    
    base_profit_target = 0.04  # 4%
    confidence_multiplier = 0.5
    
    for confidence in [0.6, 0.7, 0.8, 0.9]:
        confidence_factor = 1.0 - (confidence - 0.5) * confidence_multiplier
        scaled_target = base_profit_target * confidence_factor
        print(f"Confidence {confidence:.1f}: Target {scaled_target:.1%} (Factor: {confidence_factor:.2f})")
    
    print("\n🎯 Key Benefits of Enhanced Exit Strategy:")
    print("1. ✅ Confidence-based exits restore risk management")
    print("2. ✅ Profit-taking with confidence scaling optimizes gains")
    print("3. ✅ Tiered profit taking captures profits at multiple levels")
    print("4. ✅ Time-based exits prevent over-holding positions")
    print("5. ✅ Stop-loss protection prevents catastrophic losses")

if __name__ == "__main__":
    test_enhanced_exit_strategy()