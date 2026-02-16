# TBM Parameter Optimization Script - FINAL

## Overview

Create `scripts/compare_tbm_parameters.py` - a script to optimize Triple Barrier (TBM) parameters for better learnability, higher IC/SNR, and improved model performance.

---

# Part 1: Parameter Semantics (Normalized Units)

## Explicit Unit Separation

### Absolute Return Caps (in %)
| Parameter | Type | Range |
|-----------|------|-------|
| `tp_abs_lo_pct` | float | Minimum TP absolute % |
| `tp_abs_hi_pct` | float | Maximum TP absolute % |
| `sl_abs_lo_pct` | float | Minimum SL absolute % |
| `sl_abs_hi_pct` | float | Maximum SL absolute % |

### Multiplier Caps
| Parameter | Type | Range |
|-----------|------|-------|
| `tp_mult_lo` | float | Minimum TP multiplier |
| `tp_mult_hi` | float | Maximum TP multiplier |
| `sl_mult_lo` | float | Minimum SL multiplier |
| `sl_mult_hi` | float | Maximum SL multiplier |

## Method-Conditioned Parameters

Collapse redundant parameters into method-specific names:

| Method | Parameters Used |
|--------|-----------------|
| `tp_method='atr_mult'` | `k_tp` (multiplier) |
| `tp_method='absolute'` | `tp_abs_pct` (fixed %) |
| `tp_method='atr_norm'` | `k_tp`, `tp_base_pct` (base for normalization) |
| `sl_method='atr_mult'` | `k_sl` (multiplier) |
| `sl_method='absolute'` | `sl_abs_pct` (fixed %) |
| `sl_method='tp_pct'` | `sl_as_tp_pct` (fraction of TP) |

**Redundant parameters REMOVED:**
- `barrier_sl_base_mult` → use `k_sl` with `sl_method='atr_mult'`
- `barrier_tp_lo/hi` → use `tp_abs_lo_pct/hi_pct`
- `barrier_sl_lo/hi` → use `sl_abs_lo_pct/hi_pct`

---

# Part 2: Hierarchical Grid (Avoids Explosion)

## Stage 1: Quick Scan (Only Mode 1)

### Mode 1: atr_mult_rr (Baseline)
```
TP: k_tp × ATR% × regime_mult × horizon_scale
SL: sl_as_tp_pct × TP (RR)
```
**Grid:**
- `k_tp`: [0.8, 1.0, 1.25, 1.6, 2.0] (5 values)
- `sl_as_tp_pct`: [0.4, 0.5, 0.6, 0.7] (4 values)
- `regime_model`: ['none', 'mix'] (2 values)
- `horizon_scaling`: ['none', 'sqrt'] (2 values)

**Stage 1 Total: ~80 configs**

## Stage 2: Validation (Modes 2-4)

### Mode 2: atr_mult_independent_sl (Decoupled)
```
TP as above
SL: k_sl × ATR% × regime_mult
```
- `k_tp`: [1.0, 1.6] (2 values)
- `k_sl`: [0.5, 1.0] (2 values)
- `regime_model`: ['none', 'mix'] (2 values)
- With asymmetry: `tp_side_skew` ∈ {0.0, 0.1} (optional)

### Mode 3: absolute (Control)
```
TP = fixed %
SL = fixed % or RR
```
- Small grid for sanity check (~10 configs)

### Mode 4: atr_norm (Self-normalized)
```
TP = k_tp × (ATR% / median_ATR%) × base_TP
```
- Only run on top configs from Stage 1

### Path Dependence (Stage 2 Add-on)
Only for top 5 configs from Stage 1:
- `sl_activation_minutes`: [0, 30]
- `trail_sl_mult`: [0, 0.5]
- `tp_time_decay`: ['none', 'linear']

---

# Part 3: Regime & Asymmetry

## Regime Model (Single Structured Choice)

Instead of overlapping toggles, use single choice:
| Value | Description |
|-------|-------------|
| `'none'` | No regime adaptation |
| `'level'` | Scale by ATR level |
| `'shock'` | Scale by volatility shock |
| `'mix'` | Mix with `mix_weight` |

Applied via:
- `tp_regime_model`: Which model for TP
- `sl_regime_model`: Which model for SL (Stage 2 only - keep fixed in Stage 1)

## Asymmetry (Stage 2 Only)

Instead of separate long/short multipliers, use single skew parameter:
```
tp_side_skew ∈ {0.0, 0.1, 0.2}

k_tp_long = k_tp * (1 + skew)
k_tp_short = k_tp * (1 - skew)
```

Stage 1: Always `tp_side_skew = 0.0`

---

# Part 4: Learnability Metrics

## Dual IC Definitions

Report both to prevent "easy label" overfitting:

### IC vs Label
```
IC_label: correlation of OOF predictions vs label (−1/0/+1 or probability-weighted)
```
### IC vs Payoff
```
IC_payoff: correlation vs realized net payoff under that barrier (after fees/slip)
```

## Calibration Metrics (Cheap, High Signal)

| Metric | Description |
|--------|-------------|
| Brier Score | Probabilistic calibration |
| ECE | Expected Calibration Error |
| Monotonicity | Average payoff by prediction decile should be monotone |

## Anti-SNR Penalties

Prevent quantile filter cheating:
```
coverage = ess / ess_full  # Relative to full-sample ESS for that bucket
score *= sqrt(coverage)

# Hard gate:
worst_bucket_coverage > min_coverage_threshold
```

---

# Part 5: Filtering Rules

## Net RR (Conservative)
```
tp_net = tp - fee_pct - slip_buffer
sl_net = sl + fee_pct + slip_buffer
net_rr = tp_net / sl_net
Gate: net_rr >= min_net_rr
```

## Per-Bucket Constraints
```
min_tp_hit_rate_per_bucket >= 0.01
max_timeout_rate_per_bucket <= 0.95

# Soft fallback:
pass on at least 70% of (bucket, horizon) cells
```

## ESS Gating
Compute ESS AFTER all weights applied:
```
min_ess_events: Minimum effective sample size after weighting
```

---

# Part 6: Slice Management

## Stage 1 Slices (Minimal)
- Global
- 4 buckets: MR_long, MR_short, TF_long, TF_short

## Stage 2 Slices (Full)
- Add regime (3 states): low_vol, medium_vol, high_vol
- Add vol quintiles (5): Q1-Q5
- **CSV**: Only top-line + worst-case
- **JSON**: Full per-slice metrics (separate artifact)

---

# Part 7: Caching Strategy

## Two-Layer Cache

### Layer 1: Barrier Series
**Key:** `(method, params, vol_inputs, horizon, side)`
**Value:** TP/SL arrays

### Layer 2: TB Labels
**Key:** `(barrier_cache_key, trail_params, decay_params, delay_params)`
**Value:** Labels + hit/timeout stats

## Cache Key Serialization
```python
# Round floats to prevent cache misses
def serialize_key(params):
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in params.items()}
```

---

# Part 8: Promotion Logic

## Stage 1 → Stage 2

### Hard Gates
1. Pass all filter rules (net_rr, hit_rate, timeout, ESS)
2. IC_time_fold_min > 0 OR worst_bucket_IC > -0.05

### Pareto Criteria
1. Top-K by IC in at least one bucket
2. Not bottom-quartile on ESS or timeout

### Score
```python
stage1_score = (
    ic_snr * sqrt(ess / ess_full) 
    - 0.2 * bound_saturation 
    - 0.2 * timeout_rate
)
```

## Stage 2 Ranking

### Robust Score
```python
stage2_score = (
    0.35 * ic 
    + 0.25 * ic_snr 
    + 0.20 * sortino 
    - 0.15 * ic_std_time 
    - 0.10 * ic_std_asset
)

# Hard penalty:
if worst_bucket_IC < -0.1:
    stage2_score -= 0.5
```

---

# Part 9: Other Parameters (Complete)

## Volatility Measures
| Parameter | Values | Description |
|-----------|--------|-------------|
| `vol_measure` | ['atr_pct', 'd_atr', 'volshock'] | Primary volatility |
| `atr_window` | [24, 48, 72, 168] | ATR window (hours) |
| `base_atr_window` | [168, 336, 720] | Base ATR window |

## Quantile Filtering
| Parameter | Values | Description |
|-----------|--------|-------------|
| `use_quantile_filter` | [True, False] | Enable filtering |
| `quantile_basis` | ['vol', 'breakout', 'volume', 'trend', 'composite'] | Filter basis |
| `quantile_lo/hi` | [0.1-0.3 / 0.7-0.9] | Quantile range |
| `min_keep_fraction` | [0.3, 0.5, 0.7] | Minimum samples to keep |

## Weighting
| Parameter | Values | Description |
|-----------|--------|-------------|
| `weighting_scheme` | ['none', 'rr', 'tp_hit', 'inv_timeout', 'combined'] | Method |
| `rr_weight_power` | [0.5, 1.0, 1.5, 2.0] | RR weighting power |
| `tp_hit_weight` | [0.0, 0.5, 1.0, 1.5] | TP hit multiplier |
| `timeout_penalty` | [0.0, 0.5, 1.0] | Timeout penalty |

## Noise Buffer
| Parameter | Values | Description |
|-----------|--------|-------------|
| `sl_noise_buffer` | [True, False] | Enable min SL |
| `sl_min_abs_pct` | [0.005, 0.01, 0.015, 0.02] | Min SL % |
| `sl_min_bps` | [50, 100, 150, 200] | Min SL in BPS |
| `tp_min_abs_pct` | [0.005, 0.01, 0.015] | Min TP % |
| `tp_min_bps` | [30, 50, 75, 100] | Min TP in BPS |

## Horizon Scaling
| Parameter | Values | Description |
|-----------|--------|-------------|
| `horizon_scaling` | ['none', 'sqrt', 'power'] | Scaling type |
| `horizon_alpha` | [0.35, 0.5, 0.65] | Power for H^α |
| `horizon_base` | [4, 6, 8, 12] | Base horizon |

## Soft Saturation
| Parameter | Values | Description |
|-----------|--------|-------------|
| `soft_saturation` | [True, False] | Enable soft clipping |
| `soft_sharpness` | [5, 10, 20, 50] | Sigmoid sharpness |

## Filtering
| Parameter | Values | Description |
|-----------|--------|-------------|
| `min_net_rr` | [0.5, 0.7, 0.9, 1.1] | Min risk-reward |
| `min_tp_hit_rate` | [0.01, 0.02, 0.03, 0.05] | Min TP hit |
| `max_timeout_rate` | [0.7, 0.8, 0.9, 0.95] | Max timeout |
| `min_raw_events` | [50, 100, 200, 500] | Min events |
| `min_ess_events` | [30, 50, 100, 200] | Min ESS |
| `fee_pct` | [0.3, 0.5, 0.7] | Fee % |
| `slip_buffer` | [0.0, 0.1, 0.2] | Slip buffer % |

---

# Part 10: Example Usage

```bash
# Full optimization
python scripts/compare_tbm_parameters.py \
    --features data/features/20260214_190000 \
    --panel data/klines/20260214_190000 \
    --output reports/tbm_optimization.csv

# Quick scan
python scripts/compare_tbm_parameters.py \
    --features data/features/20260214_190000 \
    --output reports/tbm_quick.csv \
    --quick

# Resume with winners
python scripts/compare_tbm_parameters.py \
    --features data/features/20260214_190000 \
    --output reports/tbm_detailed.csv \
    --stage2 \
    --winners CONFIG001 CONFIG002 CONFIG003
```

---

# Part 11: Output Columns

## Main CSV
```
config_id, mode, k_tp, sl_method, sl_as_tp_pct, regime_model, horizon_scaling,
ic_label, ic_payoff, ic_snr, sharpe, sortino,
tp_hit_rate, sl_hit_rate, timeout_rate, 
ess, ess_full, coverage,
ic_std_time, ic_std_asset, worst_bucket_IC,
stage1_score, stage2_score
```

## Per-Bucket JSON (Separate)
```
{
  "bucket_metrics": {...},
  "regime_metrics": {...},
  "vol_quintile_metrics": {...},
  "calibration": {...}
}
```
