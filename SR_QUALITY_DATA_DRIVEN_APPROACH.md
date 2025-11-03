# SR Quality Model: Data-Driven vs Heuristic Approach

## 🎯 Your Question: What Will the Model Test Against?

**Current Problem:** The model trains on `quality_score`, which is itself a heuristic composite:

```python
quality_score = (
    bounce_strength * 0.25 +           # Normalized to 4% threshold
    hold_strength * 0.20 +             # Normalized to 20 bars
    max(trade_profit, 0) * 0.20 +      # Normalized trade P&L
    rejection_speed * 0.20 +           # Heuristic speed score
    volume_quality * 0.15              # Normalized to 2.5x
)
```

**This is circular!** Training a model to predict a heuristic score defeats the purpose of machine learning.

---

## ✅ Proper Targets (What the Model Should Actually Test Against)

### 🥇 Best: Actual Trading P&L (Most Pure)

**Target:** `realized_pnl_pct` - Real money made/lost by trading this level

**Why it's best:**
- No heuristics whatsoever
- Direct optimization for trading profit
- Model learns what actually makes money

**Example:**
```python
# For each SR level:
target = _calculate_trading_pnl(level, future_data)
# Returns: -0.01 (lost 1%) to +0.02 (made 2%)

# Train model:
model.train(X=features, y=realized_pnl_pct)
```

**What model learns:**
- If bounce is 2.8% → made 1.8% profit → 2.8% is STRONG
- If bounce is 5% → lost -1% → 5% might be FALSE SIGNAL
- No assumptions about "4% threshold" needed!

---

### 🥈 Alternative: Raw Component Metrics (Multi-Task)

**Targets:** 5 separate models for raw metrics

1. **`bounce_pct_raw`** - Actual bounce % (e.g., 0.028 = 2.8%)
2. **`bars_until_break_raw`** - Actual bars held (e.g., 25 bars)
3. **`trade_pnl_pct_raw`** - Actual trade P&L (e.g., 0.018 = 1.8%)
4. **`rejection_bar_index_raw`** - How fast rejected (0-5)
5. **`volume_ratio_raw`** - Actual volume spike (e.g., 2.8x)

**Why it's good:**
- Each model specializes
- No normalization heuristics
- Can use different targets for different strategies

---

## 📊 Real Example: Heuristic vs Data-Driven

### Level Performance:
- **Bounce:** 2.8%
- **Hold:** 25 bars
- **Trading P&L:** +1.8% (REAL MONEY)

### ❌ Heuristic Approach:

```python
# Apply fixed thresholds:
bounce_strength = min(0.028 / 0.04, 1.0)  # = 0.70 (seems weak!)
hold_strength = min(25 / 20, 1.0)          # = 1.00
quality_score = 0.70 * 0.25 + 1.00 * 0.20  # = 0.375 (mediocre)

# Model learns: This level has quality 0.375 (not great)
```

**Problems:**
1. 2.8% bounce seems "weak" (70% of 4% threshold)
2. But this level **made 1.8% profit** in reality!
3. Heuristic says mediocre, reality says excellent

### ✅ Data-Driven Approach:

```python
# Use actual trading outcome:
target = realized_pnl_pct = 0.018  # Made 1.8% profit

# Model learns: Levels with 2.8% bounce + these features → +1.8% profit
# No threshold needed! Model discovers what works.
```

**Benefits:**
1. Model learns 2.8% bounce is actually strong (led to profit)
2. Directly optimizes for trading performance
3. Discovers patterns we might miss

---

## 🔧 Implementation Steps

### Step 1: Modify Data Collection

Update `sr_quality_data_collector.py`:

```python
def _measure_level_performance(self, level, future_data, historical_data):
    """Store REAL targets, not heuristics."""
    
    # Calculate actual trading P&L (PRIMARY TARGET)
    trading_result = self._simulate_realistic_trade(
        level_type, level_price, future_data, first_hit_idx
    )
    realized_pnl_pct = trading_result['pnl_pct']  # e.g., 0.018
    
    # Get raw metrics (no normalization)
    bounce_pct_raw = weighted_bounce_pct  # e.g., 0.028
    bars_until_break_raw = bars_until_break  # e.g., 25
    
    return {
        # PRIMARY TARGET: Use this for training!
        'realized_pnl_pct': realized_pnl_pct,  # REAL MONEY
        
        # Raw components (no normalization)
        'bounce_pct_raw': bounce_pct_raw,
        'bars_until_break_raw': bars_until_break_raw,
        'trade_won': 1.0 if realized_pnl_pct > 0 else 0.0,
        
        # Keep heuristic for comparison
        'quality_score_heuristic': old_heuristic_score  # Benchmark
    }
```

### Step 2: Train on Real Target

```python
# Instead of:
model.train(
    X=training_data[feature_cols],
    y=training_data['quality_score']  # ❌ Heuristic!
)

# Use:
model.train(
    X=training_data[feature_cols],
    y=training_data['realized_pnl_pct']  # ✅ Real P&L!
)
```

### Step 3: Validate on Actual Trading

```python
# Backtest on held-out data
for date in test_dates:
    levels = test_data[test_data['date'] == date]
    
    # Predict quality using model
    predicted_quality = model.predict(levels[feature_cols])
    
    # Select top 5 levels by prediction
    top_5 = levels.nlargest(5, predicted_quality)
    
    # Measure ACTUAL P&L from these levels
    actual_pnl = top_5['realized_pnl_pct'].mean()
    
    print(f"{date}: Top 5 levels returned {actual_pnl*100:.2f}%")
```

---

## 💡 Key Insights

### Why Current Approach is Circular

1. **Heuristics in Target:**
   - `quality_score` uses fixed weights (25%, 20%, 20%, 20%, 15%)
   - Normalized to arbitrary thresholds (4%, 20 bars, 2.5x)
   
2. **Circular Learning:**
   - Model learns to predict: `0.25*bounce + 0.20*hold + ...`
   - This is just **learning to reproduce the formula**
   - Not learning from actual outcomes!

3. **Example:**
   - Heuristic says: "4% bounce = perfect"
   - Reality might be: "3% bounce is already excellent for 1h"
   - Model stuck with heuristic assumptions

### Why Data-Driven is Better

1. **Real Outcomes:**
   - Target = actual money made/lost
   - No assumptions needed
   
2. **Adaptive Thresholds:**
   - Model learns: "2.8% bounce → +1.8% profit" (strong!)
   - Model learns: "6% bounce → -1% loss" (false signal!)
   - Thresholds emerge from data, not hardcoded

3. **Non-Linear Patterns:**
   - Heuristic: Linear combination (25% + 20% + ...)
   - Data-driven: Model finds interactions
   - Example: "3% bounce + 2x volume = excellent, but 3% bounce + 1x volume = mediocre"

---

## 📝 Summary

| Aspect | Current (Heuristic) | Recommended (Data-Driven) |
|--------|-------------------|--------------------------|
| **Target** | `quality_score` (heuristic) | `realized_pnl_pct` (real P&L) |
| **Thresholds** | Fixed (4%, 20 bars) | Learned from data |
| **Weights** | Fixed (25%, 20%, 20%...) | Learned by model |
| **Optimization** | Predict heuristic score | Maximize trading profit |
| **Circular?** | ✅ Yes (learns formula) | ❌ No (learns outcomes) |
| **Adaptive?** | ❌ No (hardcoded) | ✅ Yes (data-driven) |

---

## 🚀 Next Steps

1. **Modify data collector** to store `realized_pnl_pct` as primary target
2. **Store raw metrics** (bounce_pct_raw, bars_until_break_raw, etc.)
3. **Train model** on `realized_pnl_pct` instead of `quality_score`
4. **Compare** heuristic vs data-driven on held-out test set
5. **Iterate** based on actual trading performance

---

## 📚 Files Created

1. **`multi_task_quality_model.py`** - Multi-task learning approach
2. **`raw_metrics_quality_model.py`** - Train on raw metrics without normalization
3. **`enhanced_data_collector.py`** - Store both heuristic and raw targets
4. **`proper_target_implementation.py`** - Complete implementation with realistic trading simulation
5. **`proper_targets.md`** - Detailed explanation of all target options

All files are in: `/Users/remyroche/Documents/Ares/src/tactician/sr_levels/ml_quality/`

---

## ✅ Bottom Line

**Current:** Model learns to predict `quality_score = 0.25*bounce + 0.20*hold + ...`

**Better:** Model learns to predict `realized_pnl_pct` (actual profit/loss)

**Result:** Model discovers what actually makes money, not what heuristics assume!

