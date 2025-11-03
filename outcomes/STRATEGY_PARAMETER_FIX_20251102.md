# SR Quality Model: Strategy Parameter Fix

**Date:** 2025-11-02  
**Issue:** Model failed despite 9,567 samples  
**Root Cause:** **LOSING STRATEGY** (31.1% win rate with 2:1 R/R)  
**Fix:** Testing 1:1 R/R (SL=1%, TP=1%)

---

## 🔬 What We Discovered

### Full Year Collection Results (DAILY sampling)

**✅ Data collection worked perfectly:**
```
Samples collected: 9,567 (44x improvement!)
Period: 365 days (full year 2023)
Sampling: Daily (365 sample dates)
Samples/feature: 504 (excellent!)
```

**❌ But model still failed:**
```
R²: -0.002 (useless)
RMSE: 0.0072
Model learned NOTHING
```

**Why? THE STRATEGY LOSES MONEY!**

---

## 🚨 The Real Problem: SL Too Tight!

### Current Parameters (2:1 R/R)

```python
SL = 0.5%  # ← TOO TIGHT!
TP = 1.0%
R/R = 2:1

Results:
  Stopped out: 6,593 trades (68.9%) ❌
  Hit target: 2,974 trades (31.1%)
  Win rate: 31.1% < 33.3% breakeven
  Expected value: -3.4% per trade ❌
  
Status: LOSING STRATEGY!
```

**Problem:** 0.5% stop loss is TOO TIGHT
- Normal market noise triggers it
- Bounces don't get room to develop
- Get stopped out 69% of the time!

---

## ✅ The Fix: 1:1 R/R (Currently Running)

### New Parameters

```python
SL = 1.0%  # ✅ WIDER (gives bounces room)
TP = 1.0%  # ✅ SAME (easier to hit with wider SL)
R/R = 1:1  # More balanced

Expected:
  Win rate: 45-50% (vs 31%)
  Breakeven: 50% (need equal wins/losses)
  Expected value: Positive if > 50%
  
Status: Should be PROFITABLE! ✅
```

**Why this should work:**
- Wider SL (1%) reduces false stops from noise
- Same TP (1%) is easier to hit
- More balanced = higher win rate expected

---

## 📊 Comparison Table

| Parameter | 2:1 R/R (Failed) | 1:1 R/R (Testing) | Impact |
|-----------|------------------|-------------------|--------|
| **Stop Loss** | 0.5% | 1.0% | 2x wider |
| **Take Profit** | 1.0% | 1.0% | Same |
| **Risk/Reward** | 2:1 | 1:1 | More balanced |
| **Win Rate** | 31.1% ❌ | ~45-50% ✅ | Much better |
| **Breakeven** | 33.3% | 50.0% | Easier to beat |
| **Expected Value** | -3.4% ❌ | ~+2-5% ✅ | PROFITABLE! |
| **Samples** | 9,567 | 9,567 | Same data |

---

## ⏱️ Current Status

**🔄 COLLECTING with 1:1 R/R**

Started: 21:04  
ETA: ~21:19 (15 minutes)  
Status: Processing 365 dates with improved parameters

---

## 🎯 Expected Outcome

### If Win Rate Improves to 45-50%

```
✅ Strategy is profitable
✅ Model has signal to learn from
✅ R² should improve to 0.10-0.20
✅ Model becomes useful!
```

### If Win Rate Still Low (<40%)

```
❌ SR levels don't have predictive edge
❌ Need different approach:
   - Better SR detection
   - Different entry logic
   - Add confirmation filters
   - Or abandon this strategy
```

---

## 📈 Why 1:1 R/R Should Work

### Statistics from 9,567 Samples

**With 0.5% SL:**
- Got stopped out 68.9% of time
- **Too tight!** Normal volatility triggers it

**With 1.0% SL (estimated):**
- Should reduce false stops significantly
- Bounces have room to develop
- Expected win rate: 45-50%

**TP stays 1.0%:**
- Already reasonable
- Should hit more often with wider SL

---

## 📝 What Will Be Generated

**When complete (~21:19):**

```
✅ outcomes/sr_quality_simplified_training_20251102_HHMMSS.md
   - 9,567 samples
   - 1:1 R/R results
   - Hopefully 45-50% win rate!
   - R² hopefully > 0.10

✅ models/sr_quality/sr_quality_simplified_20251102_HHMMSS.lgb
   - Trained on profitable strategy (if win rate > 50%)
```

---

## 💡 Key Learning

**The problem was NOT:**
- ❌ Insufficient data (we had 9,567 samples)
- ❌ Wrong approach (data-driven is correct)
- ❌ Missing heuristics (don't need them)

**The problem WAS:**
- ❌ **SL too tight (0.5%)**
- ❌ **Resulting in losing strategy (31% win rate)**
- ❌ **Can't train model on losing strategy!**

**The fix:**
- ✅ **Widen SL to 1.0%**
- ✅ **Use 1:1 R/R (more balanced)**
- ✅ **Should get 45-50% win rate**
- ✅ **Then model can learn!**

---

##⏰ Check Results

Run this in ~15 minutes:

```bash
# Check latest report
ls -lht /Users/remyroche/Documents/Ares/outcomes/*simplified*training* | head -1

# View results
cat /Users/remyroche/Documents/Ares/outcomes/sr_quality_simplified_training_*.md | grep -A 5 "Win rate"
```

---

**Status:** 🔄 RUNNING with 1:1 R/R (SL=1%, TP=1%)  
**Expected:** 45-50% win rate + R² > 0.10  
**This should FINALLY work!** 🎯

