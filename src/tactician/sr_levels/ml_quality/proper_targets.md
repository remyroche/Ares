# Proper Targets for Data-Driven SR Quality Models

## ❌ Current Problem: Circular Heuristics

**Current target:** `quality_score` = weighted combination of normalized metrics
```python
quality_score = bounce_strength * 0.25 + hold_strength * 0.20 + ...
```

**Issue:** Training a model to predict a heuristic score defeats the purpose of ML!

---

## ✅ Proper Targets (Ranked by Data-Drivenness)

### 🥇 Level 1: ACTUAL TRADING P&L (Most Pure)

**What:** Real money made/lost by trading this level

**Target:** `realized_pnl_pct` - Actual percentage return from trading the level

**Implementation:**
```python
def _calculate_trading_pnl(self, level, future_data, historical_data):
    """
    Simulate actual trading strategy and return real P&L.
    
    Strategy:
    1. Enter trade when level is hit
    2. Use realistic SL/TP (e.g., 1% SL, 2% TP)
    3. Exit after 10 bars if neither hit
    4. Return actual P&L %
    
    Returns:
        realized_pnl_pct: -0.01 to +0.02 (actual %)
    """
    entry_price = level.price
    
    if level.type == 'support':
        stop_loss = entry_price * 0.99
        take_profit = entry_price * 1.02
        direction = 1
    else:
        stop_loss = entry_price * 1.01
        take_profit = entry_price * 0.98
        direction = -1
    
    # Simulate trade
    for bar in future_data.iloc[:10].itertuples():
        if direction == 1:
            if bar.low <= stop_loss:
                return -0.01  # Lost 1%
            if bar.high >= take_profit:
                return +0.02  # Made 2%
        else:
            if bar.high >= stop_loss:
                return -0.01
            if bar.low <= take_profit:
                return +0.02
    
    # Exit at market
    exit_price = future_data.iloc[9].close
    return (exit_price - entry_price) / entry_price * direction

# Training:
# X = features
# y = realized_pnl_pct  ← REAL MONEY, no heuristics!
```

**Benefits:**
- No heuristics whatsoever
- Direct optimization for trading profit
- Model learns what actually makes money

**Challenges:**
- Target is noisy (single trades have variance)
- Needs many samples for stable learning
- May need risk-adjusted metric (Sharpe ratio)

---

### 🥈 Level 2: RAW COMPONENT METRICS (Good Compromise)

**What:** Train separate models for each raw performance metric

**Targets (5 separate models):**
1. `bounce_pct_raw` - Actual bounce percentage (e.g., 0.035 = 3.5%)
2. `bars_until_break_raw` - Actual bars held (e.g., 18 bars)
3. `trade_pnl_pct_raw` - Actual trade P&L (e.g., 0.012 = 1.2%)
4. `rejection_bar_index_raw` - How fast rejected (0-5)
5. `volume_ratio_raw` - Actual volume spike (e.g., 2.3x)

**Implementation:**
```python
# Multi-task learning
component_models = {
    'bounce': train_model(X, y=data['bounce_pct_raw']),
    'hold': train_model(X, y=data['bars_until_break_raw']),
    'trade': train_model(X, y=data['trade_pnl_pct_raw']),
    'speed': train_model(X, y=data['rejection_bar_index_raw']),
    'volume': train_model(X, y=data['volume_ratio_raw'])
}

# Then either:
# A) Use meta-model to combine predictions
# B) Use predictions directly for different use cases
```

**Benefits:**
- Each model specializes (better than one model for everything)
- Raw metrics = no normalization heuristics
- Can use different targets for different trading strategies

**Challenges:**
- Need to combine predictions somehow (meta-model or rules)
- Multiple models to maintain

---

### 🥉 Level 3: COMPOSITE RAW METRIC (Simpler)

**What:** Weighted sum of raw metrics, but learned weights

**Target:** Create a "true quality" metric from raw values

**Implementation:**
```python
def _calculate_true_quality(self, level, future_data, historical_data):
    """
    Calculate quality from raw metrics without normalization.
    
    Returns raw composite that model will learn to predict.
    """
    # Get raw metrics
    bounce_pct_raw = ...  # e.g., 0.035
    bars_held_raw = ...   # e.g., 18
    trade_pnl_raw = ...   # e.g., 0.012
    
    # IMPORTANT: No normalization! Just scale to similar ranges
    # Let model learn what values matter
    
    # Simple additive composite (all on 0-1 scale roughly)
    true_quality = (
        min(bounce_pct_raw / 0.10, 1.0) +  # Cap at 10% bounce
        min(bars_held_raw / 50, 1.0) +     # Cap at 50 bars
        min(trade_pnl_raw / 0.05, 1.0)     # Cap at 5% profit
    ) / 3.0  # Average
    
    return true_quality

# Training:
# y = true_quality  ← Still raw metrics, just combined
```

**Benefits:**
- Single target (simpler than multi-task)
- No arbitrary thresholds (4%, 20 bars, etc.)
- Model learns what predicts this composite

**Challenges:**
- Still has some heuristics (capping values)
- Weights are implicit in the caps

---

### 🏆 Level 4: RANKING TARGET (Best for Actual Use Case)

**What:** Train to rank levels, not predict absolute quality

**Why:** Users look at top 10 levels, not absolute scores

**Target:** Pairwise comparisons or ranking loss

**Implementation:**
```python
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

# Option A: RankNet / LambdaRank
# LightGBM supports ranking directly
train_data = lgb.Dataset(
    X, 
    label=y_realized_pnl,  # Use actual P&L
    group=[10, 10, 10, ...],  # Group by date (10 levels per day)
)

params = {
    'objective': 'lambdarank',  # Ranking objective
    'metric': 'ndcg',  # Normalized DCG
    'ndcg_eval_at': [1, 3, 5, 10]
}

model = lgb.train(params, train_data)

# Model optimized for ranking, not absolute values!
```

**Benefits:**
- Matches actual use case (ranking levels)
- More robust to label noise
- Focuses on relative quality

**Challenges:**
- Requires grouping data (levels on same date)
- Can't predict absolute quality

---

## 🎯 Recommended Approach: Combination

**Best solution:** Use multiple targets for different purposes

```python
class MultiTargetSRQualityModel:
    """
    Train on multiple targets simultaneously.
    """
    
    def __init__(self):
        self.models = {
            # Primary: Trading P&L (what we care about)
            'trading_pnl': None,
            
            # Secondary: Component metrics (diagnostic)
            'bounce_pct': None,
            'hold_bars': None,
            'rejection_speed': None,
            
            # Tertiary: Ranking (for top-K selection)
            'ranking': None
        }
    
    def train(self, data):
        # Train primary model on actual trading returns
        self.models['trading_pnl'] = train_model(
            X=data[feature_cols],
            y=data['realized_pnl_pct']  # REAL MONEY
        )
        
        # Train component models on raw metrics
        self.models['bounce_pct'] = train_model(
            X=data[feature_cols],
            y=data['bounce_pct_raw']  # RAW %
        )
        
        # Train ranking model
        self.models['ranking'] = train_ranking_model(
            X=data[feature_cols],
            y=data['realized_pnl_pct'],
            groups=data.groupby('date').size()  # Rank within date
        )
    
    def predict(self, features):
        # For production: use trading P&L prediction
        quality_scores = self.models['trading_pnl'].predict(features)
        
        # Can also get component predictions for diagnostics
        bounce_predictions = self.models['bounce_pct'].predict(features)
        
        return quality_scores
    
    def rank(self, features):
        # For top-K selection: use ranking model
        ranking_scores = self.models['ranking'].predict(features)
        return ranking_scores
```

---

## 📊 Comparison Table

| Target | Data-Driven? | Complexity | Use Case |
|--------|--------------|------------|----------|
| Current `quality_score` | ❌ No (heuristic) | Low | Baseline |
| Trading P&L | ✅✅✅ Yes | Medium | Real trading |
| Raw components | ✅✅ Mostly | High | Multi-task |
| Composite raw | ✅ Partial | Low | Simple alternative |
| Ranking | ✅✅✅ Yes | Medium | Top-K selection |

---

## 🔧 Implementation Steps

### Step 1: Modify Data Collection

Add these to `_measure_level_performance()`:

```python
return {
    # PRIMARY TARGET: Real trading P&L
    'realized_pnl_pct': self._calculate_trading_pnl(level, future_data),
    
    # RAW COMPONENTS (no normalization)
    'bounce_pct_raw': weighted_bounce_pct,  # e.g., 0.035
    'bars_until_break_raw': bars_until_break,  # e.g., 18
    'trade_pnl_pct_raw': actual_pnl,  # e.g., 0.012
    'rejection_bar_index_raw': rejection_bar,  # e.g., 2
    'volume_ratio_raw': test_volume / avg_volume,  # e.g., 2.3
    
    # METADATA
    'date': current_date,
    'level_price': level.price,
    'level_type': level.type,
    
    # Keep heuristic for comparison
    'quality_score_heuristic': quality_score  # Old approach
}
```

### Step 2: Train on Proper Targets

```python
# Option A: Train on trading P&L (most pure)
model.train(
    X=training_data[feature_cols],
    y=training_data['realized_pnl_pct']  # ← NEW TARGET
)

# Option B: Multi-task learning
for component, target in [
    ('bounce', 'bounce_pct_raw'),
    ('hold', 'bars_until_break_raw'),
    ('trade', 'trade_pnl_pct_raw')
]:
    models[component].train(
        X=training_data[feature_cols],
        y=training_data[target]
    )

# Option C: Ranking
model.train_ranking(
    X=training_data[feature_cols],
    y=training_data['realized_pnl_pct'],
    groups=training_data.groupby('date').size()
)
```

### Step 3: Validate Against Real Trading

```python
# Backtest on held-out data
predictions = model.predict(test_data[feature_cols])

# Simulate trading using predicted quality scores
for date in test_dates:
    levels_today = test_data[test_data['date'] == date]
    
    # Select top 5 by predicted quality
    top_5 = levels_today.nlargest(5, predictions)
    
    # Calculate actual P&L from these levels
    actual_pnl = top_5['realized_pnl_pct'].mean()
    
    print(f"{date}: Selected levels returned {actual_pnl*100:.2f}%")
```

---

## 💡 Key Insight

**The best target is actual trading performance, NOT a heuristic quality score.**

If a level has:
- 2% bounce (seems weak by 4% threshold)
- 15 bars hold (seems short by 20 bar threshold)
- But made +3% profit in real trading

Then it's a GOOD level! The heuristics were wrong.

Let the model learn from actual outcomes, not human assumptions.

