# SR ML Improvement - REVISED Realistic Plan

**Based on Critical Feedback - Trading Performance Focused**

---

## 🎯 Adjusted Goals

### Old (Overly Optimistic) Goals
```
❌ R² Target: 50-60%
❌ Focus: Model metrics (R², RMSE)
❌ Success: Better SHAP scores
```

### New (Realistic) Goals
```
✅ R² Target: 25-30% (ceiling for financial prediction)
✅ Focus: Trading profitability after costs
✅ Success: Sharpe ratio > 1.5, Max DD < 15%
```

---

## 📊 Reality Check: R² Expectations

| Domain | Typical R² | Why |
|--------|-----------|-----|
| Stock Returns | 2-5% | High noise, adversarial |
| Credit Risk | 20-30% | More stable relationships |
| Customer Churn | 25-40% | Behavioral patterns |
| **SR Level Quality** | **20-30%** | **Market non-stationarity** |

**Key Insight**: R² = 30% is EXCELLENT for this problem. Anything higher risks overfitting.

---

## 🚨 Critical Issue: The Real Success Metric

### The Problem With R²

```python
# Scenario 1: High R² but unprofitable
Model predicts quality = 0.85 for level at $2,500
- Spread: 0.1% = $2.50
- Entry slippage: 0.05% = $1.25
- Bounce: 0.8% = $20
- Profit after costs: $20 - $2.50 - $1.25 = $16.25 ✅

But if R:R < 2:1 due to stop loss → Unprofitable strategy

# Scenario 2: Lower R² but profitable  
Model predicts quality = 0.60 for level at $2,500
- Uses wider stops (better R:R)
- Fewer trades (lower costs)
- Higher win rate on selected trades
→ Better Sharpe ratio
```

**Conclusion**: We need **Phase 0** focused on trading simulation FIRST.

---

## 🎯 NEW Phase 0: Trading Simulation (MOST IMPORTANT)

**Goal**: Establish realistic baseline and success criteria

### Task 0.1: Build SR Level Trading Simulator

**File**: Create `src/research/sr_trading_simulator.py`

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TradingCosts:
    """Real-world transaction costs."""
    spread_pct: float = 0.001        # 0.1% spread (market orders)
    maker_fee: float = 0.0004        # 0.04% (limit orders)
    taker_fee: float = 0.0010        # 0.10% (market orders)
    slippage_pct: float = 0.0002     # 0.02% slippage
    
    @property
    def round_trip_maker(self) -> float:
        """Round-trip cost with limit orders."""
        return 2 * (self.maker_fee + self.slippage_pct)  # ~0.12%
    
    @property
    def round_trip_taker(self) -> float:
        """Round-trip cost with market orders."""
        return 2 * (self.taker_fee + self.slippage_pct) + self.spread_pct  # ~0.34%

@dataclass
class SRTradingStrategy:
    """SR-based trading strategy parameters."""
    quality_threshold: float = 0.5   # Min quality to trade
    stop_loss_atr: float = 2.0       # Stop loss in ATR units
    take_profit_atr: float = 4.0     # Take profit (2:1 R:R)
    max_holding_bars: int = 100      # Time stop
    position_size: float = 0.02      # 2% of capital per trade
    use_maker_orders: bool = True    # Try to get maker fees

class SRTradingSimulator:
    """
    Realistic SR trading simulator with transaction costs.
    
    This is the REAL test of ML model value!
    """
    
    def __init__(self, costs: TradingCosts = None):
        self.costs = costs or TradingCosts()
        self.trades: List[Dict] = []
    
    def simulate_strategy(
        self,
        sr_levels: pd.DataFrame,  # Detected levels with ML quality scores
        market_data: pd.DataFrame,  # OHLCV
        strategy: SRTradingStrategy = None
    ) -> Dict[str, Any]:
        """
        Simulate trading at SR levels with realistic costs.
        
        Returns trading performance metrics.
        """
        strategy = strategy or SRTradingStrategy()
        self.trades = []
        
        # Calculate ATR
        atr = self._calculate_atr(market_data)
        
        # Filter levels by ML quality threshold
        tradeable_levels = sr_levels[
            sr_levels['quality_score'] >= strategy.quality_threshold
        ]
        
        print(f"🎯 Tradeable levels: {len(tradeable_levels)}/{len(sr_levels)} " +
              f"(quality >= {strategy.quality_threshold})")
        
        # Simulate each level
        for _, level in tradeable_levels.iterrows():
            trade_result = self._simulate_level_trade(
                level=level,
                market_data=market_data,
                atr=atr,
                strategy=strategy
            )
            
            if trade_result:
                self.trades.append(trade_result)
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        return performance
    
    def _simulate_level_trade(
        self,
        level: pd.Series,
        market_data: pd.DataFrame,
        atr: pd.Series,
        strategy: SRTradingStrategy
    ) -> Dict[str, Any]:
        """
        Simulate a single trade at an SR level.
        
        Returns None if level never hit, trade dict if executed.
        """
        level_price = level['price']
        level_type = level['type']
        level_quality = level['quality_score']
        
        # Find when price hits level (with tolerance)
        tolerance = 0.002  # 0.2%
        
        if level_type == 'support':
            hits = market_data[market_data['low'] <= level_price * (1 + tolerance)]
        else:  # resistance
            hits = market_data[market_data['high'] >= level_price * (1 - tolerance)]
        
        if len(hits) == 0:
            return None  # Level never hit
        
        # Entry
        entry_idx = hits.index[0]
        entry_price = level_price
        entry_atr = atr.loc[entry_idx]
        
        # Calculate stops
        if level_type == 'support':
            stop_loss = entry_price - (strategy.stop_loss_atr * entry_atr)
            take_profit = entry_price + (strategy.take_profit_atr * entry_atr)
        else:  # resistance (short)
            stop_loss = entry_price + (strategy.stop_loss_atr * entry_atr)
            take_profit = entry_price - (strategy.take_profit_atr * entry_atr)
        
        # Entry cost
        cost_multiplier = self.costs.round_trip_maker if strategy.use_maker_orders else self.costs.round_trip_taker
        entry_cost = entry_price * cost_multiplier / 2  # Half of round-trip
        
        # Simulate holding period
        future_data = market_data.loc[entry_idx:].iloc[1:]  # Data after entry
        exit_reason = None
        exit_price = None
        exit_idx = None
        
        for i, (idx, row) in enumerate(future_data.iterrows()):
            # Check stop loss
            if level_type == 'support':
                if row['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    exit_idx = idx
                    break
                # Check take profit
                elif row['high'] >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    exit_idx = idx
                    break
            else:  # resistance
                if row['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    exit_idx = idx
                    break
                elif row['low'] <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    exit_idx = idx
                    break
            
            # Check time stop
            if i >= strategy.max_holding_bars:
                exit_price = row['close']
                exit_reason = 'time_stop'
                exit_idx = idx
                break
        
        # If no exit triggered, exit at end of data
        if exit_price is None:
            exit_price = future_data.iloc[-1]['close']
            exit_reason = 'end_of_data'
            exit_idx = future_data.index[-1]
        
        # Exit cost
        exit_cost = exit_price * cost_multiplier / 2  # Half of round-trip
        
        # Calculate P&L
        if level_type == 'support':
            gross_pnl = exit_price - entry_price
        else:  # resistance (short)
            gross_pnl = entry_price - exit_price
        
        net_pnl = gross_pnl - entry_cost - exit_cost
        pnl_pct = net_pnl / entry_price
        
        # Risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rrr = reward / risk if risk > 0 else 0
        
        # Holding period
        holding_bars = (exit_idx - entry_idx).seconds / 60 / 15 if hasattr((exit_idx - entry_idx), 'seconds') else 0
        
        return {
            'entry_time': entry_idx,
            'exit_time': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'level_price': level_price,
            'level_type': level_type,
            'level_quality': level_quality,
            'exit_reason': exit_reason,
            'gross_pnl': gross_pnl,
            'entry_cost': entry_cost,
            'exit_cost': exit_cost,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'holding_bars': holding_bars,
            'rrr_planned': rrr,
            'rrr_actual': abs(gross_pnl / risk) if risk > 0 else 0
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive trading performance metrics."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl_pct': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'avg_rrr': 0.0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['net_pnl'] > 0])
        losses = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = wins / total_trades
        
        # P&L stats
        avg_pnl_pct = trades_df['pnl_pct'].mean()
        std_pnl_pct = trades_df['pnl_pct'].std()
        total_return = trades_df['pnl_pct'].sum()
        
        # Sharpe ratio (annualized, assuming 15m bars)
        bars_per_year = 365 * 24 * 4  # 15-min bars in a year
        sharpe_ratio = (avg_pnl_pct * np.sqrt(bars_per_year / trades_df['holding_bars'].mean())) / std_pnl_pct if std_pnl_pct > 0 else 0
        
        # Max drawdown
        cumulative = (1 + trades_df['pnl_pct']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk-reward
        avg_rrr = trades_df['rrr_actual'].mean()
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Cost analysis
        total_costs = (trades_df['entry_cost'] + trades_df['exit_cost']).sum()
        gross_pnl_total = trades_df['gross_pnl'].sum()
        cost_ratio = total_costs / abs(gross_pnl_total) if gross_pnl_total != 0 else 0
        
        return {
            # Trade counts
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            
            # Returns
            'avg_pnl_pct': avg_pnl_pct,
            'std_pnl_pct': std_pnl_pct,
            'total_return_pct': total_return * 100,
            'best_trade_pct': trades_df['pnl_pct'].max(),
            'worst_trade_pct': trades_df['pnl_pct'].min(),
            
            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_factor': profit_factor,
            
            # R:R analysis
            'avg_rrr_planned': trades_df['rrr_planned'].mean(),
            'avg_rrr_actual': avg_rrr,
            'median_rrr_actual': trades_df['rrr_actual'].median(),
            
            # Exit analysis
            'exit_reasons': exit_reasons,
            'avg_holding_bars': trades_df['holding_bars'].mean(),
            
            # Cost analysis
            'total_costs_pct': cost_ratio * 100,
            'gross_pnl_total': gross_pnl_total,
            'net_pnl_total': trades_df['net_pnl'].sum(),
            'costs_eaten_pct': (1 - trades_df['net_pnl'].sum() / gross_pnl_total) * 100 if gross_pnl_total != 0 else 0
        }
    
    def print_performance_report(self, performance: Dict):
        """Print detailed performance report."""
        print("\n" + "="*60)
        print("  SR TRADING SIMULATION RESULTS")
        print("="*60)
        
        print(f"\n📊 TRADE STATISTICS")
        print(f"  Total Trades:        {performance['total_trades']}")
        print(f"  Winning Trades:      {performance['winning_trades']} ({performance['win_rate']*100:.1f}%)")
        print(f"  Losing Trades:       {performance['losing_trades']}")
        
        print(f"\n💰 RETURNS")
        print(f"  Total Return:        {performance['total_return_pct']:.2f}%")
        print(f"  Avg Trade P&L:       {performance['avg_pnl_pct']*100:.2f}%")
        print(f"  Best Trade:          {performance['best_trade_pct']*100:.2f}%")
        print(f"  Worst Trade:         {performance['worst_trade_pct']*100:.2f}%")
        
        print(f"\n⚡ RISK METRICS")
        print(f"  Sharpe Ratio:        {performance['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {performance['max_drawdown_pct']:.2f}%")
        print(f"  Profit Factor:       {performance['profit_factor']:.2f}")
        
        print(f"\n🎯 RISK:REWARD")
        print(f"  Avg R:R (Planned):   {performance['avg_rrr_planned']:.2f}")
        print(f"  Avg R:R (Actual):    {performance['avg_rrr_actual']:.2f}")
        print(f"  Median R:R:          {performance['median_rrr_actual']:.2f}")
        
        print(f"\n⏱️  TIMING")
        print(f"  Avg Holding:         {performance['avg_holding_bars']:.1f} bars")
        
        print(f"\n💸 COST ANALYSIS")
        print(f"  Costs vs Gross P&L:  {performance['costs_eaten_pct']:.1f}%")
        print(f"  Gross P&L:           {performance['gross_pnl_total']:.2f}")
        print(f"  Net P&L:             {performance['net_pnl_total']:.2f}")
        
        print(f"\n🚪 EXIT REASONS")
        for reason, count in performance['exit_reasons'].items():
            pct = count / performance['total_trades'] * 100
            print(f"  {reason:15s}      {count:3d} ({pct:.1f}%)")
        
        print("=" * 60)
```

### Task 0.2: Establish Baseline Performance

**Test current ML model (R² = 15.5%) in trading simulator:**

```python
# Load current SR detection results
sr_levels = pd.read_json('outcomes/sr_workflow_ETHUSDT_15m/sr_detection_ETHUSDT_15m_20251101_155519.json')
market_data = pd.read_parquet('data_cache/ETHUSDT_15m.parquet')

# Initialize simulator
simulator = SRTradingSimulator(costs=TradingCosts())

# Test current model
baseline_perf = simulator.simulate_strategy(
    sr_levels=sr_levels,
    market_data=market_data,
    strategy=SRTradingStrategy(quality_threshold=0.5)
)

simulator.print_performance_report(baseline_perf)

# SAVE BASELINE - This is what we're trying to beat!
with open('baseline_performance.json', 'w') as f:
    json.dump(baseline_perf, f, indent=2)
```

### Task 0.3: Define Success Criteria (NOT R²!)

```python
SUCCESS_CRITERIA = {
    # Primary metric: Sharpe ratio
    'sharpe_ratio': {
        'baseline': 0.5,      # Current model (assumed)
        'target': 1.5,        # Goal
        'excellent': 2.0      # Stretch goal
    },
    
    # Secondary: Max drawdown
    'max_drawdown_pct': {
        'baseline': -25,      # Current
        'target': -15,        # Goal
        'excellent': -10      # Stretch
    },
    
    # Win rate (with 2:1 R:R)
    'win_rate': {
        'baseline': 0.40,     # Current
        'target': 0.50,       # Goal (50% with 2:1 R:R = profitable)
        'excellent': 0.60     # Stretch
    },
    
    # Profit factor
    'profit_factor': {
        'baseline': 1.0,      # Break-even
        'target': 1.5,        # Goal
        'excellent': 2.0      # Stretch
    },
    
    # Cost efficiency
    'costs_eaten_pct': {
        'baseline': 40,       # 40% of gross profit eaten by costs
        'target': 20,         # Goal
        'excellent': 10       # Stretch
    }
}

def evaluate_improvement(baseline, current):
    """Check if improvements meet success criteria."""
    improvements = {}
    
    for metric, thresholds in SUCCESS_CRITERIA.items():
        baseline_val = baseline.get(metric, 0)
        current_val = current.get(metric, 0)
        target_val = thresholds['target']
        
        if 'drawdown' in metric or 'cost' in metric:
            # Lower is better
            achieved = current_val <= target_val
            improvement_pct = (baseline_val - current_val) / abs(baseline_val) * 100
        else:
            # Higher is better
            achieved = current_val >= target_val
            improvement_pct = (current_val - baseline_val) / baseline_val * 100
        
        improvements[metric] = {
            'baseline': baseline_val,
            'current': current_val,
            'target': target_val,
            'achieved': achieved,
            'improvement_pct': improvement_pct
        }
    
    return improvements
```

**CRITICAL**: R² is now just a **diagnostic metric**, not a success criterion!

---

## 📋 REVISED Implementation Phases

### Phase 0: Trading Simulation (NEW!) - 2 days

**Success = Establish realistic baseline**

- [ ] Task 0.1: Build SR trading simulator (above)
- [ ] Task 0.2: Run baseline with current model (R² = 15.5%)
- [ ] Task 0.3: Define success criteria (Sharpe > 1.5, etc.)
- [ ] Task 0.4: Sensitivity analysis (how do costs affect profitability?)

**Expected Baseline Results:**
```
Sharpe Ratio: 0.3-0.7 (current model, after costs)
Max DD: -20% to -30%
Win Rate: 35-45% (with 2:1 R:R)
Cost Impact: 30-50% of gross profit
```

---

### Phase 1: Fix Quality Score & Data Quality - 2 days

**Success = Sharpe ratio +0.3 to +0.5 improvement**

Same tasks as before, BUT:
- Measure impact on **trading Sharpe**, not R²
- R² target: 20-25% (not 28%)
- If R² improves but Sharpe doesn't → FAIL

---

### Phase 2: Enhanced Target Variable - 3-5 days

**Success = Sharpe ratio > 1.2, Win rate > 45%**

Implement multi-dimensional quality score, BUT:
- Test each component against trading performance
- Components that don't improve Sharpe → Remove
- R² target: 25-30% (ceiling)

---

### Phase 3: Feature Engineering (Realistic Version) - 1 week

**Success = Sharpe ratio > 1.5, Profit factor > 1.5**

**REVISED Feature Categories:**

#### A. Microstructure (Using Available Data)

```python
# AVAILABLE from Binance:
feature_taker_buy_ratio = taker_buy_volume / total_volume
feature_buy_pressure = taker_buy_volume / avg_volume
feature_trade_intensity = trades_count / avg_trades_count

# NOT AVAILABLE (skip):
# feature_book_imbalance ❌
# feature_tick_flow ❌
```

#### B. Volume Profile (Feasible)

```python
# Build volume profile from OHLCV
def calculate_volume_profile(data, level_price, bins=50):
    """Approximate volume profile without tick data."""
    price_range = data['high'].max() - data['low'].min()
    bin_size = price_range / bins
    
    # Assign volume to bins
    volume_profile = np.zeros(bins)
    for idx, row in data.iterrows():
        low_bin = int((row['low'] - data['low'].min()) / bin_size)
        high_bin = int((row['high'] - data['low'].min()) / bin_size)
        
        # Distribute volume across bins touched by this candle
        bins_touched = high_bin - low_bin + 1
        volume_per_bin = row['volume'] / bins_touched
        
        for b in range(low_bin, high_bin + 1):
            if 0 <= b < bins:
                volume_profile[b] += volume_per_bin
    
    # Find bin for our level
    level_bin = int((level_price - data['low'].min()) / bin_size)
    
    # Return volume at this level relative to max
    if 0 <= level_bin < bins:
        return volume_profile[level_bin] / volume_profile.max()
    return 0

feature_volume_profile_strength = calculate_volume_profile(data, level.price)
```

---

## 🎯 New Success Metrics Dashboard

Track these EVERY training run:

```python
TRACKING_METRICS = {
    # Trading Performance (PRIMARY)
    'sharpe_ratio': {'current': 0.0, 'target': 1.5, 'weight': 0.30},
    'win_rate': {'current': 0.0, 'target': 0.50, 'weight': 0.20},
    'profit_factor': {'current': 0.0, 'target': 1.5, 'weight': 0.20},
    'max_drawdown': {'current': 0.0, 'target': -15, 'weight': 0.15},
    'avg_rrr_actual': {'current': 0.0, 'target': 1.5, 'weight': 0.15},
    
    # Model Diagnostics (SECONDARY)
    'val_r2': {'current': 0.155, 'target': 0.25, 'weight': 0.0},  # Not used in score!
    'train_val_gap': {'current': 0.22, 'target': 0.08, 'weight': 0.0},
    
    # Feature Health
    'max_feature_importance': {'current': 0.64, 'target': 0.25, 'weight': 0.0},
    
    # Prediction Quality
    'prediction_calibration': {'current': 0.0, 'target': 0.80, 'weight': 0.0}
}

def calculate_overall_score(metrics):
    """
    Calculate weighted score (0-100).
    
    Only trading metrics count!
    """
    score = 0
    total_weight = 0
    
    for metric_name, metric_data in metrics.items():
        if metric_data['weight'] == 0:
            continue  # Skip diagnostic metrics
        
        current = metric_data['current']
        target = metric_data['target']
        weight = metric_data['weight']
        
        # Normalize to 0-1 (1 = met target)
        if 'drawdown' in metric_name or metric_name == 'train_val_gap':
            # Lower is better
            normalized = min(1.0, target / current) if current < 0 else 0
        else:
            # Higher is better
            normalized = min(1.0, current / target)
        
        score += normalized * weight
        total_weight += weight
    
    return (score / total_weight) * 100 if total_weight > 0 else 0
```

---

## 💡 Key Changes from Original Plan

### 1. R² Expectations Lowered
```
Old: 50-60% (unrealistic)
New: 25-30% (ceiling for financial prediction)
```

### 2. Success Metric Changed
```
Old: R² > 0.35
New: Sharpe Ratio > 1.5 (after transaction costs)
```

### 3. Phase 0 Added
```
Build realistic trading simulator BEFORE making model changes
Establish baseline performance
```

### 4. Microstructure Features Scoped
```
Old: Full order book features (not available)
New: Taker buy/sell volume (available from Binance)
```

### 5. Two-Stage Model Deprioritized
```
Old: Phase 3 priority
New: Optional - test if Phase 2 doesn't hit targets
Reason: Predicting "will be tested" = market prediction (hard problem)
```

---

## 📊 Expected Realistic Results

| Phase | Sharpe Ratio | Win Rate | R² | Status |
|-------|--------------|----------|-----|--------|
| Baseline | 0.5 | 40% | 15.5% | Current |
| Phase 1 | 0.8 | 43% | 22% | +60% Sharpe |
| Phase 2 | 1.3 | 48% | 27% | +160% Sharpe ✅ |
| Phase 3 | 1.6 | 52% | 29% | +220% Sharpe 🎯 |

**Note**: R² plateaus at ~30% (theoretical ceiling), but trading performance continues to improve through better feature engineering and risk management.

---

## 🚨 Red Flags to Watch For

### 1. R² Improves But Sharpe Doesn't
```
Symptom: Val R² goes 15% → 35%
         Sharpe stays at 0.5

Diagnosis: Model is overfitting to validation set
           OR features don't translate to trading performance

Action: Revert changes, focus on out-of-sample testing
```

### 2. Costs Eating All Profits
```
Symptom: Gross profit = 10%
         Net profit after costs = 2%
         Costs eating 80% of profit

Diagnosis: Strategy trades too frequently
           OR levels too close (frequent entries)

Action: Increase quality threshold
        Add minimum distance between levels filter
```

### 3. High Win Rate But Low Sharpe
```
Symptom: Win rate = 65%
         Sharpe = 0.6

Diagnosis: Wins are small, losses are large
           R:R ratio broken

Action: Review stop loss placement
        Check if model predicts weak bounces as high quality
```

---

## ✅ Revised Checklist

### Phase 0: Trading Simulation
- [ ] Build SR trading simulator with realistic costs
- [ ] Run baseline simulation with current model
- [ ] Document baseline Sharpe, drawdown, win rate
- [ ] Sensitivity analysis: How do costs affect results?
- [ ] Define success criteria (Sharpe > 1.5, etc.)

### Phase 1: Quick Fixes
- [ ] Fix quality scores (untouched levels = 0)
- [ ] Filter out ancient/irrelevant levels
- [ ] Remove leaky features (distance_to_current_pct)
- [ ] Add taker buy/sell volume features
- [ ] **Re-simulate trading - Did Sharpe improve?**

### Phase 2: Enhanced Target
- [ ] Implement multi-dimensional quality score
- [ ] Test each component against trading Sharpe
- [ ] Filter training data (keep only tested levels)
- [ ] **Re-simulate - Sharpe > 1.2?**

### Phase 3: Advanced Features
- [ ] Add volume profile features (using OHLCV)
- [ ] Add interaction features
- [ ] Add temporal evolution features
- [ ] **Final simulation - Sharpe > 1.5? ✅**

---

## 🎓 Final Wisdom

### What We Learned From Critique

1. **R² is a proxy, not the goal**
   - A profitable strategy with R² = 20% beats an unprofitable strategy with R² = 50%

2. **Transaction costs are critical**
   - 0.3% round-trip cost can eliminate 40-50% of gross profit
   - Model must predict levels that work AFTER costs

3. **Realistic expectations matter**
   - Financial prediction R² of 30% is excellent
   - Sharpe ratio > 1.5 is achievable and profitable

4. **Use available data wisely**
   - Binance gives us taker buy/sell volume ✅
   - Full order book not needed for good results

5. **Test what matters**
   - Trading simulator reveals true model value
   - SHAP plots are interesting but don't pay the bills

---

**Remember**: The goal is not to maximize R². The goal is to make profitable trades after transaction costs.

