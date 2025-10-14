# Economic Metrics for Regime Quality Validation

## 🎯 Purpose

The economic metrics validate whether discovered regimes represent **economically meaningful market behaviors/patterns** that justify training different ML models for each regime.

## 💰 Key Economic Questions Answered

1. **Do regimes have different risk-return profiles?** → Justifies regime-specific ML models
2. **Are regimes economically persistent enough?** → Ensures sufficient data for ML training
3. **Do regimes show different market behaviors?** → Validates regime economic significance
4. **Can regimes be exploited for trading?** → Confirms practical utility
5. **Do regimes have different risk characteristics?** → Supports risk management benefits

## 📊 Economic Metrics Categories

### 1. Risk-Return Analysis
**Purpose**: Validate that regimes have meaningfully different risk-return profiles

#### Sharpe Ratio Difference
- **What it measures**: Difference in risk-adjusted returns between regimes
- **Economic significance threshold**: >0.5 difference
- **Trading implication**: High differences → regime-specific strategies viable
- **Formula**: `max(regime_sharpe_ratios) - min(regime_sharpe_ratios)`

#### Information Ratio Difference  
- **What it measures**: Excess return per unit of tracking error vs benchmark
- **Economic significance threshold**: >0.3 difference
- **Trading implication**: High differences → strong alpha generation potential
- **Formula**: `(regime_return - benchmark_return) / tracking_error`

#### Return Separability
- **What it measures**: Annual return differences between regimes
- **Economic significance threshold**: >1% annual difference
- **Trading implication**: Significant differences justify separate ML models
- **Statistical test**: ANOVA F-test for regime return differences

#### Risk-Adjusted Return Difference
- **What it measures**: Return/volatility ratio differences
- **Economic significance threshold**: >0.5 difference
- **Trading implication**: Supports regime-specific risk management

### 2. Risk Management Metrics
**Purpose**: Validate different risk characteristics across regimes

#### Maximum Drawdown Difference
- **What it measures**: Difference in worst-case losses between regimes
- **Economic significance threshold**: >5% difference
- **Trading implication**: Significant differences require regime-specific risk controls
- **Calculation**: Compares peak-to-trough losses across regimes

#### Value at Risk (VaR) Difference
- **What it measures**: 95% VaR differences between regimes
- **Economic significance threshold**: >1% daily difference
- **Trading implication**: Different tail risks require regime-specific position sizing
- **Formula**: `95th percentile of negative returns`

### 3. Market Microstructure Economics
**Purpose**: Validate different execution and liquidity characteristics

#### Volume Profile Difference
- **What it measures**: Relative volume differences between regimes
- **Economic significance threshold**: >50% relative difference
- **Trading implication**: Different liquidity → regime-specific execution strategies
- **Calculation**: `(max_volume - min_volume) / mean_volume`

#### Liquidity Cost Difference
- **What it measures**: Transaction cost differences using spread proxy
- **Economic significance threshold**: >0.1% spread difference
- **Trading implication**: Different execution costs across regimes
- **Proxy**: `(high - low) / close` as spread estimate

### 4. Trading Economics
**Purpose**: Validate practical trading utility of regimes

#### Regime Persistence Value
- **What it measures**: How long regimes last (for ML training viability)
- **Economic significance threshold**: Average duration ≥ 10 periods
- **Trading implication**: Sufficient persistence for model training and deployment
- **Calculation**: Average regime duration vs minimum threshold

#### Transition Cost Analysis
- **What it measures**: Cost of changing positions when regimes change
- **Economic significance threshold**: <2% annual cost
- **Trading implication**: Reasonable costs support regime-based strategies
- **Formula**: `transition_frequency × (transaction_cost + market_impact)`

## 🎯 Economic Significance Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Sharpe Ratio Diff | >0.5 | Meaningful risk-adjusted performance difference |
| Information Ratio Diff | >0.3 | Significant alpha generation potential |
| Return Separability | >1% annual | Economically meaningful return differences |
| Max Drawdown Diff | >5% | Significant risk profile differences |
| VaR Difference | >1% daily | Material tail risk differences |
| Volume Profile Diff | >50% relative | Substantial liquidity differences |
| Liquidity Cost Diff | >0.1% | Meaningful execution cost differences |
| Regime Persistence | ≥10 periods | Minimum viable for ML training |
| Transition Costs | <2% annual | Acceptable strategy implementation cost |

## 📈 Interpretation Framework

### Strong Economic Foundation (≥70% metrics significant)
- **Recommendation**: Proceed with regime-specific ML models
- **Implication**: Regimes represent economically distinct market behaviors
- **Strategy**: Focus on most significant dimensions for model training

### Moderate Economic Foundation (40-70% metrics significant)
- **Recommendation**: Selective regime-based modeling
- **Implication**: Some economic differences exist
- **Strategy**: Focus on economically significant metrics only

### Weak Economic Foundation (<40% metrics significant)
- **Recommendation**: Consider single-model approach or regime redefinition
- **Implication**: Limited economic justification for separate models
- **Strategy**: Investigate alternative clustering methods

## 🔄 Integration with ML Model Training

### Economic Validation → ML Strategy Decision Tree

```
Economic Validation Results
├── Strong Foundation → Train separate ML models per regime
├── Moderate Foundation → Selective regime modeling
└── Weak Foundation → Single model or regime redefinition
```

### Regime-Specific Model Training Justification

**Train separate models when**:
- Return separability >1% annual
- Sharpe ratio difference >0.5
- Regime persistence ≥10 periods
- Transition costs <2% annual

**Use single model when**:
- Economic metrics below thresholds
- High transition costs
- Low regime persistence
- Minimal risk-return differences

## 📊 Example Economic Report Output

```
# Economic Validation Report for Market Regimes

## Executive Summary
- Total Metrics Evaluated: 9
- Economically Significant: 6
- Economic Significance Rate: 66.7%

## Key Economic Findings
✅ Sharpe Ratio Difference: 0.73 (Economically significant)
   - Trading Implication: High differences suggest regime-specific strategies viable

✅ Return Separability: 2.1% annual range (Economically significant)  
   - Trading Implication: Return differences justify regime-specific ML models

✅ Regime Persistence Value: 2.3 (Average duration: 23 periods)
   - Trading Implication: Sufficient persistence for ML model training

## Recommendations for ML Model Training
⚠️ Moderate Economic Foundation
- Some economic significance detected
- Consider selective regime-based modeling
- Focus on most significant metrics
```

## 🎯 Usage in Framework

```python
from src.regime.clusters import RegimeValidationMetrics

# Initialize validator
validator = RegimeValidationMetrics()

# Run economic validation
economic_results = validator.validate_economic_significance(
    market_data, regime_labels
)

# Check economic significance
if economic_results['economic_summary']['overall_economic_quality'] == 'strong':
    print("✅ Strong economic foundation - proceed with regime-specific ML models")
    # Train separate models per regime
else:
    print("⚠️ Consider single model approach")
    # Use single model or redefine regimes
```

This economic validation ensures that your regime discovery produces **economically meaningful patterns** that justify the complexity of training different ML models for each regime, rather than just statistically interesting clusters.