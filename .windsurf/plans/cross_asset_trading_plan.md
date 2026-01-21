# Cross-Asset Trading Architecture Plan

This plan transforms the single-asset Ares trading system into a cross-asset framework following de Prado's causal methodology while maintaining existing single-asset capabilities.

## Current Architecture Analysis

The current system is single-asset focused:
- **Layer 2**: Base models with causal surprises per asset
- **Layer 3**: Meta-models using IRM on ranging/trending regimes  
- **Layer 4**: ExtraTrees position sizer per asset
- **Layer 5**: Portfolio optimization (basic)
- **Chaser**: Single-asset residual learner

Data is processed per-symbol with separate pipelines for BTC, ETH, etc.

## Cross-Asset Transformation Strategy

### Phase 1: Data Structure Overhaul

**Panel Data Format**
- Transform from per-asset DataFrames to stacked "long" format
- Primary Key: [Timestamp, Ticker] 
- Target: Next-period return/label per ticker per time
- Enable cross-asset feature engineering

**Asset Universe**
```python
ASSETS = {
    'BTC': {'category': 'benchmark'},
    'ETH': {'category': 'benchmark'}, 
    'SOL': {'category': 'L1'},
    'AVAX': {'category': 'L1'},
    'DOT': {'category': 'L1'},
    'NEAR': {'category': 'L1'},
    'ADA': {'category': 'L1'},
    'BNB': {'category': 'L1'},
    'LINK': {'category': 'DeFi'},
    'UNI': {'category': 'DeFi'},
    'MKR': {'category': 'DeFi'},
    'AAVE': {'category': 'DeFi'}
}
```

### Phase 2: Layer 2 Cross-Asset Enhancements

**New Cross-Asset Detectors**
1. **VPIN Spillover Surprise**
   ```python
   VPIN_spillover = VPIN_asset - E[VPIN_asset | VPIN_BTC, vol_mkt, vol_asset, time_of_day]
   ```

2. **Error Correction Term (ECT)**
   ```python
   ECT = Current_Price - (Alpha + Beta * BTC_Price)
   # Activate only when Johansen Test confirms cointegration
   ```

3. **Beta-Drift Surprise**
   ```python
   Beta_Drift_Surprise = (β_t|t − β_t|t−1) / sqrt(Pβ_t|t−1)
   ```

**Implementation**: Extend `label_based_layer_2.py` with cross-asset feature engineering module.

### Phase 3: Layer 3 Cross-Asset Meta Model

**Environment Shift**
- Move IRM from ranging/trending to per-asset environment detection
- Add cross-sectional features

**New Features**
- BTC Dominance (BTC.D) Delta
- Cross-Sectional Volatility (Dispersion)  
- Beta Deviation
- Internal Dispersion: std(Returns_15_assets)
- Market Return (Rm)
- Beta-Standard-Error

**Target Enhancement**
```python
# Market-Adjusted Threshold
y_i = Return_i - (β_i * Return_market)

# FracDiff Features
Residual_t = log(PAsset_t) - (β+ * 1_Rm>0 + β- * 1_Rm<0) * log(PMarket_t)
Feature = FracDiff((Δlog(PAsset)/Δlog(PMarket)) - β_rolling)
```

### Phase 4: Layer 4 Cross-Asset Position Sizing

**Top-K Confidence Selection**
- Train ExtraTrees on all assets conjointly
- Only accept trades with confidence in top 2 across all assets
- Generate independent confidence per asset
- Scale models independently then together

**Risk Controls**
- Limit Value at Risk through top-K selection
- Increase risk:reward ratio by high-confidence filtering

### Phase 5: Enhanced Portfolio Optimization

**Cross-Asset Constraints**
1. No two assets with correlation > 0.7
2. Cap aggregate beta exposure  
3. Max 33% Kelly exposure open at any time
4. Position exit rules:
   - Trailing profit hit
   - Confidence(New) > Confidence(Current) + 0.2

**Percentile-Based Selection**
- Standardize per-asset scores into cross-asset percentiles
- Use rolling calibration set for normalization
- Select top-K by percentile, not raw probability

### Phase 6: Cross-Asset Chaser

**New Features**
- Cross-Asset Lead/Lag (averaged, not pairwise)
- Relative Volume (RV) Clusters
- Distance from "Peer Mean"
- Beta-Relative Volatility
- Beta Convexity: Up-Beta - Down-Beta

**Implementation**: Extend `layer2_5_chaser.py` with cross-asset residual learning.

### Phase 7: Ticker Embeddings & Validation

**Ticker ID Integration**
- Use Ticker ID as feature for ExtraTrees (Student) but not Huber (Teacher)
- Demean features per ticker before student training
- Force student to learn why ID matters, not just bias

**Validation Strategy**
- **Leave-One-Asset-Out (LOAO)** cross-validation
- Train on 14 assets, test on 15th
- If model fails on unseen asset, features are too specific

**Feature Engineering**
```python
# Demeaned Features
Feature_demeaned = Feature_ticker - Feature_mean(sector)

# Label Encoding for Trees
ticker_encoded = categorical_ticker_id  # 0-14

# Time-Based Splitting (always)
train_split = "2023-01-01"  # Never by row
```

### Phase 8: Implementation Architecture

**New Modules**
1. `cross_asset_data_processor.py` - Panel data transformation
2. `cross_asset_features.py` - VPIN, ECT, Beta-Drift calculations
3. `cross_asset_validation.py` - LOAO cross-validation
4. `ticker_embedding_utils.py` - ID encoding and demeaning
5. `cross_asset_portfolio_optimizer.py` - Enhanced Layer 5

**Modified Files**
- `label_based_layer_2.py` - Add cross-asset detectors
- `label_based_layer_3.py` - Per-asset IRM + cross-sectional features  
- `label_based_layer_4.py` - Top-K confidence selection
- `layer2_5_chaser.py` - Cross-asset residual features
- `label_based_pipeline.py` - Panel data support

**Configuration**
- New `cross_asset_config.yaml` with asset universe and parameters
- Per-asset vs cross-asset feature flags
- LOAO validation settings

### Phase 9: Backward Compatibility

Maintain single-asset mode through configuration:
```python
config = {
    'mode': 'single_asset',  # or 'cross_asset'
    'assets': ['ETHUSDT'],
    'cross_asset_features': False
}
```

### Phase 10: Testing & Validation

**Unit Tests**
- Panel data transformation integrity
- Cross-asset feature calculations
- LOAO validation correctness

**Integration Tests**  
- End-to-end cross-asset pipeline
- Performance vs single-asset baseline
- Feature importance analysis

**Live Trading Rollout**
- Paper trading with cross-asset signals
- Risk management validation
- Performance monitoring

## Core Architectural Risks & Mitigations

### 1.1 Regime Leakage via BTC Conditioning

**Risk**: BTC becomes latent regime classifier causing double-counting and LOAO failure

**Mitigation**: Replace single BTC anchor with Market State Vector:
```python
MarketState_t = PCA([BTC_ret, ETH_ret, BTC_vol, ETH_vol, BTC.D, funding_rates])
```
- VPIN spillover conditioned on MarketState_t
- ECT activation gated by state-specific cointegration  
- Beta drift estimated conditional on state clusters

### 1.2 Layer-Specific Feedback Controls

**Layer 2 Cross-Asset Surprises**:
- VPIN: Use quantile regression for conditional expectation to avoid mean-reversion bias
- Directional asymmetry: Separate buy/sell VPIN spillovers
- ECT: Add half-life filter and rolling rank stability requirements

**Layer 3 Meta-Model**:
- Drop raw ticker ID entirely at Layer 3
- Enforce gradient alignment penalty across assets
- Require ∂ŷ/∂feature_i invariance across asset environments

**Layer 4 Position Sizing**:
- Entropy filter: Reject flat confidence distributions (H > threshold)
- Inter-asset confidence decay: adj_conf_i = conf_i / sqrt(N_active_assets)

### 1.3 Enhanced Portfolio Constraints

**Beyond correlation > 0.7**:
- Tail correlation constraint (bottom 5% returns)
- Beta clustering penalty
- Fractional Kelly per regime: Kelly_alloc = Kelly * Regime_Confidence

### 1.4 Cross-Asset Chaser Enhancement

**Peer Residual Momentum**:
```python
PeerResidual_i = Return_i - mean(Return_peer_group)
Signal = EWMA(PeerResidual_i)
```

### 1.5 Robust Validation Framework

**Beyond LOAO**:
- Leave-One-Sector-Out (LOSO): Train on DeFi+Benchmark, test on L1
- Synthetic Asset Test: Linear combinations to prevent overfitting

### 1.6 Implementation Order Adjustment

**Recommended Sequence**:
1. Phase 1 – Panel Data
2. Phase 7 – LOAO Framework (early!)
3. Phase 2 – Cross-Asset Layer 2  
4. Phase 3 – Meta-Model
5. Phase 4 – Position Sizing
6. Phase 5 – Portfolio
7. Phase 6 – Chaser

**Rationale**: Validation constraints should shape feature design, not vice versa

## Success Criteria

1. **Performance**: Cross-asset Sharpe > single-asset baseline
2. **Diversification**: Portfolio correlation < 0.7
3. **Robustness**: LOAO + LOSO validation passes for all assets
4. **Stability**: Feature importance consistent across assets
5. **Risk Control**: VaR within limits, drawdown controlled
6. **Regime Integrity**: No BTC anchor leakage, MarketState vector stable

## Timeline Estimate (Revised)

- **Phase 1**: 1 week (Panel Data + LOAO Framework)
- **Phase 2**: 2 weeks (Cross-Asset Layer 2 with mitigations)
- **Phase 3**: 2 weeks (Meta-Model with gradient alignment)
- **Phase 4**: 2 weeks (Position Sizing + entropy filters)
- **Phase 5**: 1 week (Portfolio + enhanced constraints)
- **Phase 6**: 1 week (Chaser + peer residual momentum)

**Total**: ~9 weeks with stronger architectural foundations

This plan maintains the existing de Prado causal framework while adding sophisticated cross-asset capabilities for improved risk-adjusted returns.
