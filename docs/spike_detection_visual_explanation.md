# Spike Detection - Visual Explanation

## How Spike Detection Works

### Condition 1: Deviation from Baseline

```
Price Chart:
           
    110 |              * ← SPIKE (s_t)
        |             /|\
    105 |   *---*---*   *---*---*  ← Rolling Median Baseline
        |                         
    100 |                         
        +---------------------------
         t-5  t-3  t-1  t+1  t+3
```

**Detection**: `|s_t - median(s_{t-1..t-N})| > threshold`
- Current price (110) deviates significantly from baseline (105)
- Threshold = k × recent_std (e.g., 3 × 1.5 = 4.5)
- Deviation = |110 - 105| = 5 > 4.5 ✓ SPIKE CONDITION 1 MET

### Condition 2: Direction Reversal

```
Movement Direction:
           
    110 |              * ← SPIKE (s_t)
        |           ↗ | ↘
        |         ↗   |   ↘
    105 |   *---*     |     *---*
        |             |         
        |      UP    SPIKE   DOWN
        +---------------------------
           t-2  t-1   t   t+1  t+2
```

**Detection**: `sign(s_t - s_{t-1}) != sign(s_{t+1} - s_t)`
- Movement before spike: UP (↗) → `s_t - s_{t-1} > 0` → sign = +1
- Movement after spike: DOWN (↘) → `s_{t+1} - s_t < 0` → sign = -1
- Signs differ (+1 ≠ -1) ✓ SPIKE CONDITION 2 MET

**Result**: BOTH conditions met → This is a SPIKE → Apply correction

### Spike Correction

```
BEFORE Correction:
           
    110 |              * ← SPIKE (s_t)
        |             /|\
    105 |   *---*---*   *---*---*
        |                         
        +---------------------------
           t-2  t-1   t  t+1  t+2

Correction Formula (3-bar average):
corrected_price = (prev_price + spike_price + next_price) / 3
                = (105 + 110 + 105) / 3
                = 106.67

Why 3-bar average?
- More conservative: partially preserves spike (may contain real signal)
- Smooths noise while retaining some price movement information
- Better than 2-bar average which completely discards spike value

AFTER Correction:
           
    110 |              
        |             *  ← Smoothed (106.67)
    105 |   *---*---*   *---*---*  ← Spike smoothed, slight bump preserved
        |                         
        +---------------------------
           t-2  t-1   t  t+1  t+2
```

### Trend Preservation

```
GENUINE TREND (NOT a spike):
           
    115 |                     * ← Continues up (t+1)
        |                   ↗
    110 |              * ← Movement up (t)
        |            ↗ 
    105 |   *---*---*  
        |                         
        +---------------------------
           t-2  t-1   t  t+1  t+2

Movement Check:
- Movement before: UP → s_t - s_{t-1} > 0 → sign = +1
- Movement after: UP → s_{t+1} - s_t > 0 → sign = +1
- Signs same (+1 = +1) ✗ CONDITION 2 NOT MET

Result: NOT a spike → Trend preserved
```

## Real-World Examples

### Example 1: Flash Crash Spike

```
Market Data:
Time    | Price  | Analysis
--------|--------|------------------------------------------
10:00   | 100.50 | Normal
10:15   | 100.60 | Normal trend
10:30   | 95.00  | ← SPIKE (flash crash) -5.6%
10:45   | 100.70 | Returns to normal
11:00   | 100.80 | Trend continues

Spike Detection:
- Deviation: |95 - 100.6| = 5.6 > threshold ✓
- Direction: DOWN then UP (reversal) ✓
- Action: Correct to (100.60 + 95.00 + 100.70)/3 = 98.77
- Result: Spike smoothed but some downward movement preserved
```

### Example 2: Genuine Breakout (NOT a spike)

```
Market Data:
Time    | Price  | Analysis
--------|--------|------------------------------------------
10:00   | 100.50 | Ranging
10:15   | 100.60 | Building pressure
10:30   | 105.00 | ← BREAKOUT +4.4%
10:45   | 105.50 | Continues up
11:00   | 106.00 | Trend established

Spike Detection:
- Deviation: |105 - 100.6| = 4.4 > threshold ✓
- Direction: UP then UP (no reversal) ✗
- Action: NO correction - genuine trend preserved
```

### Example 3: Exchange Glitch

```
Market Data:
Time    | Price  | Analysis
--------|--------|------------------------------------------
10:00   | 100.50 | Normal
10:15   | 100.60 | Normal
10:30   | 110.00 | ← GLITCH +9.4% (API error)
10:45   | 100.70 | Correct price restored
11:00   | 100.80 | Normal continues

Spike Detection:
- Deviation: |110 - 100.6| = 9.4 > threshold ✓
- Direction: UP then DOWN (reversal) ✓
- Action: Correct to (100.60 + 110.00 + 100.70)/3 = 103.77
- Result: Glitch smoothed to 103.77 (partway between normal and spike)
- Spike magnitude: 9.4% (large - may indicate data issue)
```

## Detection Algorithm Flow

```
┌─────────────────────────────────────┐
│ Input: Market Data (OHLCV)         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Rolling Median Baseline   │
│ median(s_{t-1..t-N})                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Calculate Dynamic Threshold         │
│ threshold = k × rolling_std         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Check Condition 1:                  │
│ |s_t - median| > threshold?         │
└──────────────┬──────────────────────┘
               │
          ┌────┴────┐
          │         │
         NO        YES
          │         │
          ▼         ▼
     [No Spike] ┌─────────────────────────────┐
                │ Check Condition 2:          │
                │ Direction reversal?         │
                │ sign(↑) != sign(↓)?         │
                └──────────────┬──────────────┘
                               │
                          ┌────┴────┐
                          │         │
                         NO        YES
                          │         │
                          ▼         ▼
                  [Genuine Trend]  [SPIKE!]
                  [Preserve]       [Correct]
                                      │
                                      ▼
                            ┌──────────────────────────┐
                            │ Calculate 3-bar average: │
                            │ (prev + spike + next)/3  │
                            └─────────┬────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Replace spike    │
                            │ with correction  │
                            └─────────┬────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │ Track statistics │
                            └──────────────────┘
```

## Statistics Interpretation

### Good Signal Quality
```
🔍 Spike Detection Results:
   • Spikes detected: 42
   • Spikes corrected: 41
   • Correction rate: 97.6%  ← High correction rate ✓
   • Avg spike magnitude: 0.34%  ← Small magnitude ✓
   • Max spike magnitude: 1.89%  ← Reasonable max ✓
```
**Interpretation**: Clean data with occasional noise. System working well.

### Potential Data Quality Issues
```
🔍 Spike Detection Results:
   • Spikes detected: 512
   • Spikes corrected: 485
   • Correction rate: 94.7%
   • Avg spike magnitude: 2.34%  ← Large magnitude ⚠️
   • Max spike magnitude: 8.45%  ← Very large max 🚨
```
**Interpretation**: High spike rate and magnitude may indicate:
- Exchange API issues
- Data feed problems
- Network connectivity issues
- Need to review data source

## Parameter Tuning Guide

### Sensitivity Control: `threshold_multiplier`

```
Lower threshold (more sensitive):
threshold_multiplier = 2.0

    110 |    *     Catches smaller spikes
        |   /|\    More spikes detected
    105 |--*-+-*-- Baseline ± 2σ
        |     
Range: 100-110 (±5% from baseline)

Higher threshold (less sensitive):
threshold_multiplier = 4.0

    115 |    *     Only catches large spikes
        |   /|\    Fewer spikes detected
    105 |--*-+-*-- Baseline ± 4σ
        |     
Range: 95-115 (±10% from baseline)
```

**Recommendation**:
- **Crypto (volatile)**: 2.5 - 3.0
- **Stocks (less volatile)**: 3.0 - 4.0
- **Forex (very stable)**: 4.0 - 5.0

### Baseline Stability: `lookback_window`

```
Short window (N=5):
    *---*---*---*---*  ← Recent baseline
                   ^^^ More responsive to recent changes

Long window (N=20):
*---*---*---*---*---*---*---*---*---*  ← Stable baseline
                                   ^^^ More stable, less noise
```

**Recommendation**:
- **High-frequency (1m, 5m)**: 10-15 bars
- **Medium-frequency (15m, 1h)**: 10-20 bars
- **Low-frequency (4h, 1d)**: 20-30 bars

## Integration with Opportunity Detection

```
┌──────────────────────────────────────────────────────────┐
│                    Data Processing Flow                    │
└──────────────────────────────────────────────────────────┘

1. Load Market Data
   ↓
   OHLCV data with potential spikes

2. Spike Detection & Correction ← NEW STEP
   ↓
   Cleaned OHLCV data (spikes removed)

3. Volatility-Aware Labeling
   ↓
   Labels generated on clean data
   (more accurate thresholds)

4. Opportunity Detection
   ↓
   Trading opportunities identified
   (fewer false positives)

5. Quality Filtering
   ↓
   High-quality signals
   (improved signal-to-noise ratio)
```

## Benefits Visualization

```
WITHOUT Spike Detection:
Price    Opportunities    Analysis
110 |    *     ⚠️        False positive (spike)
105 |---*-*---  ✓        Real opportunities
100 |    *     ⚠️        False positive (spike)
     
Result: 4 opportunities (2 real, 2 false) → 50% accuracy

WITH Spike Detection (3-bar smoothing):
Price    Opportunities    Analysis
110 |                    
107 |    *              Spike smoothed to 106.67 (3-bar avg)
105 |---*-*---  ✓        Real opportunities detected
100 |                    Spike smoothed
     
Result: 2 opportunities (2 real, 0 false) → 100% accuracy
Note: Small bump at 107 from smoothed spike is acceptable (may contain signal)
```

## Conclusion

The spike detection system uses a sophisticated dual-condition approach to:
1. ✅ Identify anomalous price movements (deviation)
2. ✅ Distinguish spikes from trends (direction reversal)
3. ✅ Correct noise while preserving signals (3-bar average)
4. ✅ Improve downstream opportunity detection
5. ✅ Provide comprehensive monitoring statistics

**Key Principles**: 
- *If it looks like a spike and acts like a spike, smooth it with 3-bar average*
- *If it continues as a trend, preserve it completely*
- *Partial preservation: spike may contain real signal, so don't discard entirely*

