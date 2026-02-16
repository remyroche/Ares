# Candidate Selection Threshold Comparison Plan

## Objective

Compare raw return vs ATR-normalized return for candidate selection ranking to determine which approach yields better "learnability" (predictive signal in training data).

## Background

### Current Approach: Raw Return Ranking

The current candidate selection in [`select_trade_candidates_vectorized()`](extreme_price_movements/candidates.py:159) uses raw 24h returns:

```python
pct = 0.08  # train_extreme_pct_hourly
k = max(1, int(n_cols * pct))  # top K + bottom K symbols
# Ranks by raw ret24h
```

**How it works:**
- Ranks all symbols by raw 24h return (`ret24h`)
- Selects top K (best performers) + bottom K (worst performers)
- With 600 symbols and pct=0.08: K=48, so 96 candidates per timestamp

**Limitations:**
- Same threshold for all assets regardless of volatility profile
- BTC (low volatility, ~2% daily ATR) and DOGE (high volatility, ~10% daily ATR) treated identically
- A 5% move in BTC (2.5× ATR) is less likely to be selected than a 6% move in DOGE (0.6× ATR)
- High-vol assets dominate the candidate pool

### Proposed Approach: ATR-Normalized Return Ranking

Normalize returns by each asset's ATR before ranking:

```python
# atr_pct already computed per asset in features.py:1011
feats["atr_pct"] = pick_by_rv(atr_fast, atr_base, atr_slow)

# ATR-normalized return (how many ATRs did price move?)
ret24h_atr_norm = ret24h / atr_pct  # e.g., 5% move / 2% ATR = 2.5 ATRs
```

**How it works:**
- Compute `ret24h_atr_norm = ret24h / atr_pct` for each asset
- Rank all symbols by ATR-normalized return
- Select top K + bottom K (same 6%, 7%, 8% structure)
- Same candidate count, but different composition

**Advantages:**
- Rebalances selection across volatility regimes
- A 5% move in BTC (2.5× ATR) now ranks higher than 6% in DOGE (0.6× ATR)
- More economically meaningful: "extreme" = move relative to typical volatility
- Low-vol assets get fair representation in candidate pool

---

## Comparison Framework

### Test Configurations

| Config ID | Type | Ranking Metric | Description |
|-----------|------|----------------|-------------|
| F08 | Fixed | `ret24h` | Top/bottom 8% by raw return |
| F07 | Fixed | `ret24h` | Top/bottom 7% by raw return |
| F06 | Fixed | `ret24h` | Top/bottom 6% by raw return |
| A08 | ATR | `ret24h / atr_pct` | Top/bottom 8% by ATR-normalized return |
| A07 | ATR | `ret24h / atr_pct` | Top/bottom 7% by ATR-normalized return |
| A06 | ATR | `ret24h / atr_pct` | Top/bottom 6% by ATR-normalized return |
| VW08 | Vol-Weight | `|ret24h| × vol_combined` | Top/bottom 8% by vol-weighted return |
| VW07 | Vol-Weight | `|ret24h| × vol_combined` | Top/bottom 7% by vol-weighted return |
| VW06 | Vol-Weight | `|ret24h| × vol_combined` | Top/bottom 6% by vol-weighted return |

Where `vol_combined = (rvol_z + volu_z) / 2`

### Metrics for "Learnability"

#### Primary Metrics

1. **Information Coefficient (IC)**
   - Correlation between prediction and actual return
   - Measured on OOF predictions
   - Higher = more predictive signal

2. **Label Separation (KS Statistic)**
   - Kolmogorov-Smirnov distance between positive/negative label distributions
   - Higher = labels more distinguishable

3. **Signal-to-Noise Ratio**
   - `SNR = |mean_ret_pos - mean_ret_neg| / sqrt(var_pos + var_neg)`
   - Higher = cleaner signal

#### Secondary Metrics

4. **Candidate Count Statistics**
   - Mean/median candidates per timestamp
   - Cross-sectional distribution (how many symbols qualify)

5. **Class Balance**
   - Positive/negative label ratio
   - Target: 30-40% positive (matches quantile threshold)

6. **Feature-Target Correlation**
   - Mean |IC| of top 20 features with target
   - Higher = features more informative

7. **Sharpe Ratio of Selected Candidates**
   - Mean return / std return of candidates
   - Higher = better quality candidates

---

## Implementation Plan

### Phase 1: Lightweight Analysis Script (No Full Pipeline)

Create a standalone analysis script that loads existing feature data and computes comparison metrics.

**File:** `scripts/compare_candidate_thresholds.py`

```python
# Pseudocode structure
def compare_thresholds(feature_path, panel_path):
    # Load existing data
    feats = load_features(feature_path)
    panel = load_panel(panel_path)
    
    # Fixed percentage candidates
    for pct in [0.08, 0.07, 0.06]:
        candidates = select_fixed_pct(feats, pct)
        metrics = compute_learnability_metrics(candidates, feats, panel)
        results.append(("fixed", pct, metrics))
    
    # ATR-based candidates
    atr_pct = feats["atr_pct"]
    ret24h = feats["ret24h"]
    for mult in [1.5, 2.0, 2.5]:
        candidates = select_atr_based(ret24h, atr_pct, mult)
        metrics = compute_learnability_metrics(candidates, feats, panel)
        results.append(("atr", mult, metrics))
    
    return results
```

**Key Functions to Implement:**

1. `select_fixed_pct(feats, pct)` - Existing logic from candidates.py
2. `select_atr_based(ret24h, atr_pct, mult)` - New ATR-based selection
3. `compute_learnability_metrics(candidates, feats, panel)` - Metric computation

### Phase 2: Run Analysis

```bash
# Run on existing feature data (no re-computation needed)
python scripts/compare_candidate_thresholds.py \
    --features data/features/20260214_190000 \
    --panel data/klines \
    --output reports/candidate_threshold_comparison.csv
```

### Phase 3: Interpret Results

Compare metrics across configurations:

| Config | Candidates/Day | IC | KS | SNR | Sharpe | Class Balance |
|--------|----------------|----|----|-----|--------|---------------|
| F08 | ~96 | ? | ? | ? | ? | ? |
| F07 | ~84 | ? | ? | ? | ? | ? |
| F06 | ~72 | ? | ? | ? | ? | ? |
| A08 | ~96 | ? | ? | ? | ? | ? |
| A07 | ~84 | ? | ? | ? | ? | ? |
| A06 | ~72 | ? | ? | ? | ? | ? |
| VW08 | ~96 | ? | ? | ? | ? | ? |
| VW07 | ~84 | ? | ? | ? | ? | ? |
| VW06 | ~72 | ? | ? | ? | ? | ? |

**Decision Criteria:**
- If ATR-based shows higher IC/KS/SNR with similar candidate counts: **Adopt ATR**
- If volume-weighted shows higher IC/KS/SNR: **Adopt Volume-Weighted**
- If fixed percentage shows better metrics: **Keep current approach**
- If similar: Consider hybrid approaches

---

## Technical Details

### ATR-Based Selection Logic

```python
def select_atr_cross_sectional(feats, ret_col="ret24h", pct=0.08):
    """
    Cross-sectional ATR-normalized return ranking.
    
    metric = ret24h / atr_pct  # How many ATRs did price move?
    
    Selects top/bottom K by ATR-normalized return (stable candidate count).
    """
    ret = feats[ret_col]
    atr_pct = feats["atr_pct"]
    
    # ATR-normalized return (how many ATRs did price move?)
    metric = ret / atr_pct
    
    n = metric.shape[1]
    k = max(1, int(n * pct))
    
    # Rank and select top/bottom K (same logic as fixed)
    ranks = metric.rank(axis=1, method='first')
    valid_counts = metric.notna().sum(axis=1)
    vc = valid_counts.values[:, np.newaxis]
    r = ranks.values
    
    mask_arr = (r > (vc - k)) | (r <= k)
    mask_arr = mask_arr & metric.notna().values
    mask_arr[valid_counts.values < k, :] = False
    
    return pd.DataFrame(mask_arr, index=ret.index, columns=ret.columns)
```

### Volume-Weighted Selection Logic

```python
def select_volume_weighted(feats, ret_col="ret24h", pct=0.08):
    """
    Rank candidates by volume-weighted return.
    
    vol_combined = (rvol_z + volu_z) / 2  # Average of hour-of-day and 30-day volume z-scores
    weighted_return = |ret24h| × vol_combined.clip(lower=0)
    
    A 5% move with 2× normal volume ranks equal to:
    - A 10% move with 1× normal volume
    - A 2.5% move with 4× normal volume
    """
    ret = feats[ret_col]
    rvol_z = feats["rvol_z"]  # Hour-of-day adjusted volume z-score
    volu_z = feats["volu_z"]  # 30-day volume z-score
    
    # Combined volume z-score (average of hour-of-day and 30-day)
    vol_combined = (rvol_z + volu_z) / 2
    
    # Volume-weighted return (amplify high-volume moves)
    weighted_ret = ret.abs() * vol_combined.clip(lower=0)
    # Preserve sign for direction
    metric = weighted_ret * np.sign(ret)
    
    n = weighted_ret.shape[1]
    k = max(1, int(n * pct))
    
    # Rank and select top/bottom K (same as fixed/ATR)
    ranks = metric.rank(axis=1, method='first')
    valid_counts = metric.notna().sum(axis=1)
    vc = valid_counts.values[:, np.newaxis]
    r = ranks.values
    
    mask_arr = (r > (vc - k)) | (r <= k)
    mask_arr = mask_arr & metric.notna().values
    mask_arr[valid_counts.values < k, :] = False
    
    return pd.DataFrame(mask_arr, index=ret.index, columns=ret.columns)
```

**Rationale:**
- `rvol_z` captures intraday patterns (volume vs same hour historically)
- `volu_z` captures longer-term volume anomalies (30-day window)
- Averaging provides balanced sensitivity to both timeframes
- Volume amplifies the significance of price moves

### Unified Cross-Sectional Selection

All three approaches share the same selection structure:
1. Compute a ranking metric per asset per timestamp
2. Rank all assets cross-sectionally
3. Select top K + bottom K (stable count)

The only difference is **what defines "extreme"**:

| Aspect | Fixed | ATR | Volume-Weighted |
|--------|-------|-----|-----------------|
| Ranking metric | `ret24h` | `ret24h / atr_pct` | `|ret24h| × vol_combined` |
| Cross-sectional | Yes | Yes | Yes |
| Candidate count | Stable (~96/day at 8%) | Stable (~96/day at 8%) | Stable (~96/day at 8%) |
| Asset-specific | No | Yes (via ATR) | Yes (via volume z-score) |
| Captures | Price extremes | Volatility-adjusted extremes | Volume-confirmed extremes |

### Metric Computation

```python
def compute_learnability_metrics(candidates, feats, panel):
    """
    Compute learnability metrics for a candidate selection method.
    """
    # Get candidate returns
    ret24h = feats["ret24h"]
    candidate_returns = ret24h.where(candidates).stack()
    
    # 1. Information Coefficient (using ret24h as proxy)
    # In real training, this would be OOF prediction correlation
    ic = candidate_returns.mean() / candidate_returns.std()
    
    # 2. Label Separation (KS statistic)
    # Simulate labels using quantile threshold
    labels = (ret24h > ret24h.quantile(0.65)).astype(int)
    pos_dist = ret24h.where((candidates) & (labels == 1)).dropna()
    neg_dist = ret24h.where((candidates) & (labels == 0)).dropna()
    ks_stat = scipy.stats.ks_2samp(pos_dist, neg_dist).statistic
    
    # 3. Signal-to-Noise Ratio
    snr = abs(pos_dist.mean() - neg_dist.mean()) / np.sqrt(pos_dist.var() + neg_dist.var())
    
    # 4. Candidate count
    n_candidates = candidates.sum(axis=1).mean()
    
    # 5. Class balance
    label_rate = labels.where(candidates).mean()
    
    # 6. Feature-target correlation (top features)
    feature_ics = []
    for feat_name in ["vol_z", "ret1h", "rsi", "mkt_rv_ratio"]:
        if feat_name in feats:
            feat = feats[feat_name].where(candidates)
            ic_feat = feat.corrwith(ret24h).mean()
            feature_ics.append(abs(ic_feat))
    mean_feat_ic = np.mean(feature_ics) if feature_ics else 0
    
    # 7. Sharpe ratio
    sharpe = candidate_returns.mean() / candidate_returns.std() * np.sqrt(24 * 365)
    
    return {
        "ic": ic,
        "ks": ks_stat,
        "snr": snr,
        "n_candidates": n_candidates,
        "label_rate": label_rate,
        "mean_feat_ic": mean_feat_ic,
        "sharpe": sharpe,
    }
```

---

## Expected Outcomes

### Hypothesis 1: ATR-Based Better for High-Vol Assets

- ATR thresholds should better capture "meaningful" moves in volatile assets
- Fixed percentage may over-select from high-vol assets (any small move qualifies)

### Hypothesis 2: Fixed Percentage More Stable

- Fixed percentage guarantees consistent candidate counts
- ATR may have high variance (many candidates in volatile periods, few in calm)

### Hypothesis 3: Hybrid Approach Optimal

- Use ATR for assets with `atr_pct > median(atr_pct)` (high-vol)
- Use fixed percentage for low-vol assets
- Best of both worlds

---

## Next Steps

1. **Create analysis script** (`scripts/compare_candidate_thresholds.py`)
2. **Run on existing feature data** (no pipeline re-run needed)
3. **Review results** and decide on approach
4. **If ATR better:** Modify `select_trade_candidates_vectorized()` to support ATR mode
5. **If fixed better:** Document rationale and keep current approach
6. **If hybrid:** Implement hybrid selection logic

---

## Files to Modify (If ATR Adopted)

1. [`extreme_price_movements/config.py`](extreme_price_movements/config.py:200-201)
   - Add `candidate_selection_mode: "fixed" | "atr" | "hybrid"`
   - Add `candidate_atr_mult: float` (default 2.0)

2. [`extreme_price_movements/candidates.py`](extreme_price_movements/candidates.py:159-252)
   - Add `select_atr_based()` function
   - Modify `select_trade_candidates_vectorized()` to support mode selection

3. [`extreme_price_movements/training.py`](extreme_price_movements/training.py:42-44)
   - Read new config parameters
   - Pass mode to candidate selection

---

## Timeline

| Step | Description | Mode |
|------|-------------|------|
| 1 | Create analysis script | Code |
| 2 | Run comparison analysis | Code |
| 3 | Review results with user | Ask |
| 4 | Implement chosen approach | Code |
| 5 | Test with full pipeline run | Code |

---

## Appendix: ATR Feature Details

From [`extreme_price_movements/features.py:1009-1012`](extreme_price_movements/features.py:1009-1012):

```python
atr_fast = atr_percent(h, l, c, max(2, int(cfg["atr_n"] * 0.5)))
atr_slow = atr_percent(h, l, c, int(cfg["atr_n"] * 2))
feats["atr_pct"] = pick_by_rv(atr_fast, atr_base, atr_slow)
```

- `atr_n` default: 14 (periods)
- `atr_pct` = ATR / Close (percentage)
- Typical values: 0.01-0.05 for major pairs, 0.05-0.15 for altcoins

---

## Appendix: Scaled ATR Function

From [`extreme_price_movements/optimise_tpsl_ratio.py:274-306`](extreme_price_movements/optimise_tpsl_ratio.py:274-306):

```python
def scaled_atr_pct(atr_pct, z, atr_base_pct, *, z_max=3.0, lo=0.03, hi=0.06):
    """
    ATR-informed, shock-scaled, bounded barrier percent.
    
    Parameters:
    -----------
    atr_pct : float or array
        Current ATR as percentage
    z : float or array
        ATR z-score (current ATR vs historical)
    atr_base_pct : float
        Baseline ATR percentage
    z_max : float
        Maximum z-score scaling
    lo, hi : float
        Output bounds (3% to 6%)
    
    Returns:
    --------
    scaled_pct : float or array
        Dynamically scaled barrier percentage
    """
    shock = np.clip(z, 0.0, z_max)
    a = (hi / atr_base_pct - 1.0) / z_max
    raw = atr_pct * (1.0 + a * shock)
    return np.clip(raw, lo, hi)
```

This function could be used to create regime-aware ATR thresholds (higher threshold in high-vol regimes).