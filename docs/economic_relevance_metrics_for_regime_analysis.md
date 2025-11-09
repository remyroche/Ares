# Comprehensive Economic Relevance Metrics for Regime Ensemble Training Systems

## Table of Contents
1. [Introduction](#introduction)
2. [Return-Based Metrics](#return-based-metrics)
3. [Risk Metrics](#risk-metrics)
4. [Statistical Measures](#statistical-measures)
5. [Performance Persistence Metrics](#performance-persistence-metrics)
6. [Regime-Specific Implementation Considerations](#regime-specific-implementation-considerations)
7. [Interpretation Guidelines](#interpretation-guidelines)
8. [Implementation Best Practices](#implementation-best-practices)
9. [Code Examples and Integration](#code-examples-and-integration)

## Introduction

This document provides comprehensive economic relevance metrics for evaluating financial market regimes in regime ensemble training systems. These metrics are designed to assess the economic significance and practical applicability of identified market regimes, going beyond purely statistical measures to focus on actionable financial insights.

### Purpose of Regime Economic Analysis

- **Validate Economic Relevance**: Ensure that identified regimes have meaningful economic implications
- **Guide Trading Strategy Development**: Identify which regimes are most favorable for specific trading approaches
- **Risk Management**: Understand risk profiles across different market conditions
- **Performance Attribution**: Attribute performance to specific market environments
- **Model Selection**: Choose appropriate models for different regime conditions

### Key Principles

1. **Economic Significance**: Metrics must reflect real economic value, not just statistical significance
2. **Regime Specificity**: Calculations should account for regime-specific characteristics
3. **Comparability**: Metrics should enable meaningful comparisons across regimes
4. **Practical Applicability**: Results should be actionable for trading and investment decisions

## Return-Based Metrics

### 1.1 Average Returns per Regime

**Definition and Purpose:**
Average returns per regime measure the mean performance of assets or strategies during specific market regimes, providing insight into which regimes are most favorable for trading.

**Mathematical Formula:**
```
μ_r = (1/n) * Σ(r_i)
```
Where:
- μ_r = Average return for the regime
- n = Number of observations in the regime
- r_i = Individual returns in the regime

**Regime-Specific Implementation:**
```python
def average_return_per_regime(returns, regime_labels):
    """
    Calculate average returns for each market regime
    
    Args:
        returns: Array of returns
        regime_labels: Array of regime classifications
    
    Returns:
        Dictionary with average returns per regime
    """
    regime_returns = {}
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns[regime] = np.mean(returns[regime_mask])
    return regime_returns
```

**Interpretation Guidelines:**
- Positive values indicate profitable regimes
- Compare across regimes to identify most favorable market conditions
- Values > 0.5% daily are considered strong for liquid markets
- Negative values may indicate regimes to avoid or hedge against

### 1.2 Return Distribution Characteristics

**Definition and Purpose:**
Return distribution characteristics analyze the shape, spread, and tail behavior of returns within each regime, revealing risk-return profiles.

**Mathematical Formulas:**

**Standard Deviation:**
```
σ = √[(1/(n-1)) * Σ(r_i - μ_r)²]
```

**Skewness:**
```
S = [(1/n) * Σ((r_i - μ_r)/σ)³] / [(1/n) * Σ((r_i - μ_r)/σ)²]^(3/2)
```

**Kurtosis:**
```
K = [(1/n) * Σ((r_i - μ_r)/σ)⁴] / [(1/n) * Σ((r_i - μ_r)/σ)²]²
```

**Regime-Specific Implementation:**
```python
def return_distribution_metrics(returns, regime_labels):
    """
    Calculate comprehensive return distribution metrics per regime
    """
    metrics = {}
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        metrics[regime] = {
            'mean': np.mean(regime_returns),
            'std': np.std(regime_returns, ddof=1),
            'skewness': scipy.stats.skew(regime_returns),
            'kurtosis': scipy.stats.kurtosis(regime_returns, fisher=True),
            'percentiles': {
                '5th': np.percentile(regime_returns, 5),
                '25th': np.percentile(regime_returns, 25),
                '75th': np.percentile(regime_returns, 75),
                '95th': np.percentile(regime_returns, 95)
            }
        }
    return metrics
```

**Interpretation Guidelines:**
- **Skewness**: Positive (>0.5) indicates right-skewed returns (more extreme positive outcomes)
- **Kurtosis**: Values >3 indicate fat tails (higher probability of extreme events)
- **Standard Deviation**: Higher values indicate more volatile regimes
- **Percentile Analysis**: Wide spreads between 5th and 95th percentiles indicate high uncertainty

### 1.3 Risk-Adjusted Returns

**Definition and Purpose:**
Risk-adjusted returns measure performance relative to the risk taken, providing a normalized comparison across different regimes.

**Mathematical Formula:**
```
RAR = μ_r / σ_r
```
Where:
- RAR = Risk-adjusted return
- μ_r = Average return for the regime
- σ_r = Standard deviation of returns for the regime

**Regime-Specific Implementation:**
```python
def risk_adjusted_returns(returns, regime_labels, risk_free_rate=0.02):
    """
    Calculate risk-adjusted returns per regime
    """
    metrics = {}
    daily_rf_rate = risk_free_rate / 252  # Convert annual to daily
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        excess_returns = regime_returns - daily_rf_rate
        
        if np.std(regime_returns) > 0:
            metrics[regime] = np.mean(excess_returns) / np.std(regime_returns)
        else:
            metrics[regime] = 0
    
    return metrics
```

**Interpretation Guidelines:**
- Values > 1.0 indicate excellent risk-adjusted performance
- Values between 0.5-1.0 indicate good performance
- Values < 0.5 indicate poor risk-adjusted returns
- Negative values indicate underperformance relative to risk

## Risk Metrics

### 2.1 Sharpe Ratio per Regime

**Definition and Purpose:**
The Sharpe ratio measures risk-adjusted return by comparing excess returns to volatility, specifically for each market regime.

**Mathematical Formula:**
```
SR = (μ_r - r_f) / σ_r
```
Where:
- SR = Sharpe ratio for the regime
- μ_r = Average return for the regime
- r_f = Risk-free rate
- σ_r = Standard deviation of returns for the regime

**Regime-Specific Implementation:**
```python
def sharpe_ratio_per_regime(returns, regime_labels, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio for each regime
    """
    metrics = {}
    daily_rf_rate = risk_free_rate / 252  # Convert annual to daily
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        excess_returns = regime_returns - daily_rf_rate
        
        if np.std(regime_returns) > 0:
            metrics[regime] = np.mean(excess_returns) / np.std(regime_returns) * np.sqrt(252)
        else:
            metrics[regime] = 0
    
    return metrics
```

**Interpretation Guidelines:**
- Values > 1.5 indicate excellent risk-adjusted performance
- Values 1.0-1.5 indicate good performance
- Values 0.5-1.0 indicate moderate performance
- Values < 0.5 indicate poor risk-adjusted returns
- Negative values indicate underperformance relative to risk-free rate

### 2.2 Sortino Ratio per Regime

**Definition and Purpose:**
The Sortino ratio measures risk-adjusted return using downside deviation instead of total volatility, focusing on harmful volatility.

**Mathematical Formula:**
```
Sortino = (μ_r - r_f) / σ_d
```
Where:
- σ_d = Downside deviation = √[(1/n) * Σ(min(r_i - r_f, 0))²]

**Regime-Specific Implementation:**
```python
def sortino_ratio_per_regime(returns, regime_labels, risk_free_rate=0.02):
    """
    Calculate Sortino ratio for each regime
    """
    metrics = {}
    daily_rf_rate = risk_free_rate / 252
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        excess_returns = regime_returns - daily_rf_rate
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            metrics[regime] = np.mean(excess_returns) * np.sqrt(252) / downside_deviation
        else:
            metrics[regime] = float('inf') if np.mean(excess_returns) > 0 else 0
    
    return metrics
```

**Interpretation Guidelines:**
- Values > 2.0 indicate excellent downside risk management
- Values 1.0-2.0 indicate good downside protection
- Values < 1.0 indicate poor downside risk management
- Higher Sortino than Sharpe indicates asymmetric return distribution

### 2.3 Volatility per Regime (Realized and Implied)

**Definition and Purpose:**
Volatility measures the degree of variation in returns, with realized volatility based on historical data and implied volatility derived from option prices.

**Mathematical Formulas:**

**Realized Volatility:**
```
σ_realized = √[(1/(n-1)) * Σ(r_i - μ_r)²] * √(252)
```

**Implied Volatility (from options):**
```
σ_implied = Black-Scholes implied volatility from option prices
```

**Regime-Specific Implementation:**
```python
def volatility_per_regime(returns, regime_labels, implied_vols=None):
    """
    Calculate volatility metrics for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Realized volatility (annualized)
        realized_vol = np.std(regime_returns, ddof=1) * np.sqrt(252)
        
        metrics[regime] = {
            'realized_volatility': realized_vol,
            'daily_volatility': np.std(regime_returns, ddof=1),
            'volatility_regime': classify_volatility_regime(realized_vol)
        }
        
        # Add implied volatility if available
        if implied_vols is not None:
            regime_implied = implied_vols[regime_mask]
            metrics[regime]['implied_volatility'] = np.mean(regime_implied)
            metrics[regime]['vol_risk_premium'] = metrics[regime]['implied_volatility'] - realized_vol
    
    return metrics

def classify_volatility_regime(vol):
    """Classify volatility regime based on annualized volatility"""
    if vol < 0.15:
        return "low_volatility"
    elif vol < 0.25:
        return "normal_volatility"
    elif vol < 0.35:
        return "high_volatility"
    else:
        return "extreme_volatility"
```

**Interpretation Guidelines:**
- **Low Volatility** (<15% annual): Stable market conditions, lower risk premium
- **Normal Volatility** (15-25%): Typical market conditions
- **High Volatility** (25-35%): Uncertain market conditions, higher risk premium
- **Extreme Volatility** (>35%): Crisis conditions, very high risk premium
- **Vol Risk Premium**: Positive values indicate market expects higher future volatility

### 2.4 Maximum Drawdown per Regime

**Definition and Purpose:**
Maximum drawdown measures the largest peak-to-trough decline in portfolio value during a specific regime.

**Mathematical Formula:**
```
MDD = max(Peak_t - Trough_t) / Peak_t
```
Where:
- Peak_t = Maximum cumulative return up to time t
- Trough_t = Minimum cumulative return after the peak

**Regime-Specific Implementation:**
```python
def max_drawdown_per_regime(returns, regime_labels):
    """
    Calculate maximum drawdown metrics for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + regime_returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        metrics[regime] = {
            'max_drawdown': np.min(drawdown),
            'max_drawdown_duration': max_drawdown_duration(drawdown),
            'average_drawdown': np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
        }
    
    return metrics

def max_drawdown_duration(drawdown_series):
    """Calculate the maximum duration of drawdown in periods"""
    is_drawdown = drawdown_series < 0
    duration = 0
    max_duration = 0
    
    for dd in is_drawdown:
        if dd:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    
    return max_duration
```

**Interpretation Guidelines:**
- Values < -5% indicate low drawdown regimes
- Values -5% to -15% indicate moderate drawdown
- Values -15% to -25% indicate high drawdown regimes
- Values < -25% indicate extreme drawdown (crisis) regimes
- Duration > 50 trading days indicates prolonged recovery periods

### 2.5 Calmar Ratio per Regime

**Definition and Purpose:**
The Calmar ratio measures risk-adjusted return using maximum drawdown as the risk measure, providing insight into return quality relative to worst-case losses.

**Mathematical Formula:**
```
Calmar = μ_annual / |MDD|
```
Where:
- μ_annual = Annualized average return
- MDD = Maximum drawdown (absolute value)

**Regime-Specific Implementation:**
```python
def calmar_ratio_per_regime(returns, regime_labels):
    """
    Calculate Calmar ratio for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Annualized return
        annual_return = np.mean(regime_returns) * 252
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + regime_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = np.min(drawdown)
        
        # Calmar ratio
        if max_dd != 0:
            metrics[regime] = annual_return / abs(max_dd)
        else:
            metrics[regime] = float('inf') if annual_return > 0 else 0
    
    return metrics
```

**Interpretation Guidelines:**
- Values > 3.0 indicate excellent risk-adjusted performance
- Values 1.5-3.0 indicate good performance
- Values 0.5-1.5 indicate moderate performance
- Values < 0.5 indicate poor risk-adjusted returns
- Negative values indicate losses during the regime

### 2.6 Value at Risk (VaR) per Regime

**Definition and Purpose:**
Value at Risk measures the potential loss in value of a risky asset over a defined period for a given confidence interval.

**Mathematical Formulas:**

**Parametric VaR (Normal Distribution):**
```
VaR_α = μ_r + σ_r * Φ^(-1)(α)
```

**Historical VaR:**
```
VaR_α = Percentile(r_i, α)
```

Where:
- α = Confidence level (e.g., 0.05 for 95% VaR)
- Φ^(-1) = Inverse standard normal CDF

**Regime-Specific Implementation:**
```python
def var_per_regime(returns, regime_labels, confidence_levels=[0.01, 0.05, 0.1]):
    """
    Calculate Value at Risk for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        metrics[regime] = {}
        
        for alpha in confidence_levels:
            # Historical VaR
            historical_var = np.percentile(regime_returns, alpha * 100)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(regime_returns)
            std_return = np.std(regime_returns)
            parametric_var = mean_return + std_return * norm.ppf(alpha)
            
            metrics[regime][f'VaR_{int(alpha*100)}'] = {
                'historical': historical_var,
                'parametric': parametric_var,
                'scaled_annual': historical_var * np.sqrt(252)
            }
    
    return metrics
```

**Interpretation Guidelines:**
- VaR represents the minimum expected loss with given confidence
- More negative values indicate higher risk
- Compare VaR across regimes to identify highest-risk periods
- Historical VaR > Parametric VaR indicates fat tails
- Annualized VaR helps compare across different timeframes

### 2.7 Expected Shortfall (ES) per Regime

**Definition and Purpose:**
Expected Shortfall (Conditional VaR) measures the expected loss beyond the VaR threshold, providing information about tail risk.

**Mathematical Formula:**
```
ES_α = E[r | r ≤ VaR_α] = (1/α) * ∫[0,α] VaR_u du
```

**Regime-Specific Implementation:**
```python
def expected_shortfall_per_regime(returns, regime_labels, confidence_levels=[0.01, 0.05, 0.1]):
    """
    Calculate Expected Shortfall for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        metrics[regime] = {}
        
        for alpha in confidence_levels:
            # Calculate VaR first
            var = np.percentile(regime_returns, alpha * 100)
            
            # Expected Shortfall (average of returns below VaR)
            tail_returns = regime_returns[regime_returns <= var]
            es = np.mean(tail_returns) if len(tail_returns) > 0 else var
            
            metrics[regime][f'ES_{int(alpha*100)}'] = {
                'expected_shortfall': es,
                'var': var,
                'tail_ratio': es / var if var != 0 else 1,
                'scaled_annual': es * np.sqrt(252)
            }
    
    return metrics
```

**Interpretation Guidelines:**
- ES is always more negative than VaR for the same confidence level
- Higher ES/VaR ratio indicates fatter tails
- Values < -5% (daily) indicate significant tail risk
- Compare ES across regimes to identify worst-case scenarios
- Useful for stress testing and risk management

## Statistical Measures

### 3.1 Per-Regime Coefficient of Variation (CV) for Returns

**Definition and Purpose:**
The coefficient of variation measures the relative variability of returns compared to their mean, providing a normalized measure of dispersion across regimes.

**Mathematical Formula:**
```
CV = σ_r / |μ_r|
```
Where:
- CV = Coefficient of variation
- σ_r = Standard deviation of returns for the regime
- μ_r = Mean return for the regime

**Regime-Specific Implementation:**
```python
def cv_per_regime(returns, regime_labels):
    """
    Calculate coefficient of variation for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        mean_return = np.mean(regime_returns)
        std_return = np.std(regime_returns, ddof=1)
        
        # Handle zero mean case
        if mean_return != 0:
            cv = std_return / abs(mean_return)
        else:
            cv = float('inf') if std_return > 0 else 0
        
        metrics[regime] = {
            'cv': cv,
            'mean_return': mean_return,
            'std_return': std_return,
            'volatility_normalized': cv < 1.0  # Lower CV indicates better risk-adjusted consistency
        }
    
    return metrics
```

**Interpretation Guidelines:**
- CV < 0.5: Low relative variability, consistent returns
- CV 0.5-1.0: Moderate relative variability
- CV 1.0-2.0: High relative variability
- CV > 2.0: Very high relative variability, unreliable returns
- Lower CV indicates more predictable performance within the regime

### 3.2 Within-Regime CV for Key Economic Variables

**Definition and Purpose:**
Within-regime CV measures the internal consistency of key economic variables during specific market conditions, helping identify stable vs. chaotic regimes.

**Mathematical Formula:**
```
CV_within = σ_variable / |μ_variable|
```

**Regime-Specific Implementation:**
```python
def within_regime_cv(economic_variables, regime_labels, variable_names):
    """
    Calculate CV for economic variables within each regime
    
    Args:
        economic_variables: DataFrame with economic variables
        regime_labels: Array of regime classifications
        variable_names: List of variable names to analyze
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_data = economic_variables[regime_mask]
        
        metrics[regime] = {}
        
        for var in variable_names:
            if var in regime_data.columns:
                var_values = regime_data[var].dropna()
                if len(var_values) > 1:
                    mean_val = np.mean(var_values)
                    std_val = np.std(var_values, ddof=1)
                    
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                    else:
                        cv = float('inf') if std_val > 0 else 0
                    
                    metrics[regime][var] = {
                        'cv': cv,
                        'mean': mean_val,
                        'std': std_val,
                        'stability_score': 1 / (1 + cv)  # Higher = more stable
                    }
    
    return metrics

# Example usage for key economic variables
key_variables = [
    'interest_rate', 'inflation_rate', 'gdp_growth', 'unemployment_rate',
    'volatility_index', 'credit_spread', 'term_spread', 'market_sentiment'
]
```

**Interpretation Guidelines:**
- CV < 0.1: Very stable variable within regime
- CV 0.1-0.3: Moderately stable
- CV 0.3-0.5: High variability
- CV > 0.5: Very unstable, regime may be poorly defined
- Stability score > 0.7 indicates well-defined regime characteristics

### 3.3 Between-Regime CV Comparisons

**Definition and Purpose:**
Between-regime CV compares the variability of the same economic variable across different regimes, identifying which variables best distinguish between regimes.

**Mathematical Formula:**
```
CV_between = σ_regime_means / |μ_regime_means|
```

**Regime-Specific Implementation:**
```python
def between_regime_cv(economic_variables, regime_labels, variable_names):
    """
    Calculate CV for economic variables across different regimes
    """
    metrics = {}
    
    for var in variable_names:
        if var in economic_variables.columns:
            regime_means = []
            regime_stds = []
            
            for regime in np.unique(regime_labels):
                regime_mask = regime_labels == regime
                regime_data = economic_variables[regime_mask][var].dropna()
                
                if len(regime_data) > 0:
                    regime_means.append(np.mean(regime_data))
                    regime_stds.append(np.std(regime_data, ddof=1))
            
            if len(regime_means) > 1:
                # CV of regime means (between-regime variability)
                mean_of_means = np.mean(regime_means)
                std_of_means = np.std(regime_means, ddof=1)
                
                if mean_of_means != 0:
                    cv_between = std_of_means / abs(mean_of_means)
                else:
                    cv_between = float('inf') if std_of_means > 0 else 0
                
                # Average within-regime CV
                within_cvs = []
                for i, regime_mean in enumerate(regime_means):
                    if regime_mean != 0:
                        within_cv = regime_stds[i] / abs(regime_mean)
                        within_cvs.append(within_cv)
                
                avg_within_cv = np.mean(within_cvs) if within_cvs else 0
                
                metrics[var] = {
                    'cv_between': cv_between,
                    'avg_cv_within': avg_within_cv,
                    'discrimination_ratio': cv_between / (avg_within_cv + 1e-8),
                    'regime_means': regime_means,
                    'regime_stds': regime_stds
                }
    
    return metrics
```

**Interpretation Guidelines:**
- High CV_between (>0.5): Variable differs significantly between regimes (good discriminator)
- Low CV_between (<0.2): Variable similar across regimes (poor discriminator)
- Discrimination ratio > 2.0: Excellent regime discrimination capability
- Variables with high discrimination ratio are best for regime identification

### 3.4 Skewness and Kurtosis per Regime

**Definition and Purpose:**
Skewness measures the asymmetry of return distributions, while kurtosis measures the thickness of tails, both crucial for understanding risk profiles within regimes.

**Mathematical Formulas:**

**Skewness:**
```
S = E[(r_i - μ_r)³] / σ_r³ = [(1/n) * Σ((r_i - μ_r)/σ_r)³]
```

**Kurtosis (Fisher):**
```
K = E[(r_i - μ_r)⁴] / σ_r⁴ - 3 = [(1/n) * Σ((r_i - μ_r)/σ_r)⁴] - 3
```

**Regime-Specific Implementation:**
```python
def skewness_kurtosis_per_regime(returns, regime_labels):
    """
    Calculate skewness and kurtosis for returns in each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) > 3:  # Need minimum samples for meaningful stats
            skew = scipy.stats.skew(regime_returns)
            kurt = scipy.stats.kurtosis(regime_returns, fisher=True)  # Fisher kurtosis (excess)
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = scipy.stats.jarque_bera(regime_returns)
            
            # Anderson-Darling test for normality
            ad_stat, ad_critical_values, ad_significance_levels = scipy.stats.anderson(regime_returns, 'norm')
            
            metrics[regime] = {
                'skewness': skew,
                'kurtosis': kurt,
                'jarque_bera': {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'is_normal': jb_pvalue > 0.05
                },
                'anderson_darling': {
                    'statistic': ad_stat,
                    'critical_values': ad_critical_values,
                    'significance_levels': ad_significance_levels
                },
                'distribution_type': classify_distribution(skew, kurt),
                'tail_risk': assess_tail_risk(kurt),
                'asymmetry_risk': assess_asymmetry_risk(skew)
            }
    
    return metrics

def classify_distribution(skew, kurt):
    """Classify the return distribution based on skewness and kurtosis"""
    if abs(skew) < 0.5 and abs(kurt) < 0.5:
        return "approximately_normal"
    elif skew > 0.5 and kurt < 0.5:
        return "moderately_right_skewed"
    elif skew < -0.5 and kurt < 0.5:
        return "moderately_left_skewed"
    elif abs(skew) < 0.5 and kurt > 0.5:
        return "symmetric_fat_tailed"
    elif skew > 0.5 and kurt > 0.5:
        return "right_skewed_fat_tailed"
    elif skew < -0.5 and kurt > 0.5:
        return "left_skewed_fat_tailed"
    else:
        return "complex_distribution"

def assess_tail_risk(kurt):
    """Assess tail risk based on kurtosis"""
    if kurt < 0.5:
        return "low_tail_risk"
    elif kurt < 1.5:
        return "moderate_tail_risk"
    elif kurt < 3.0:
        return "high_tail_risk"
    else:
        return "extreme_tail_risk"

def assess_asymmetry_risk(skew):
    """Assess asymmetry risk based on skewness"""
    if skew < -1.0:
        return "high_negative_asymmetry"
    elif skew < -0.5:
        return "moderate_negative_asymmetry"
    elif skew < 0.5:
        return "approximately_symmetric"
    elif skew < 1.0:
        return "moderate_positive_asymmetry"
    else:
        return "high_positive_asymmetry"
```

**Interpretation Guidelines:**

**Skewness:**
- 0 ± 0.5: Approximately symmetric
- > 0.5: Right-skewed (more extreme positive returns)
- < -0.5: Left-skewed (more extreme negative returns)
- > 1.0: Strong right skew (favorable for long positions)
- < -1.0: Strong left skew (unfavorable, high crash risk)

**Kurtosis:**
- 0 ± 0.5: Normal tails
- > 0.5: Fat tails (higher probability of extreme events)
- > 1.0: Very fat tails (significant tail risk)
- > 2.0: Extreme fat tails (crisis-prone regime)

**Normality Tests:**
- Jarque-Bera p-value > 0.05: Cannot reject normality
- Anderson-Darling statistic < critical value: Cannot reject normality
- Non-normal distributions require alternative risk models

## Performance Persistence Metrics

### 4.1 Hit Rate per Regime

**Definition and Purpose:**
Hit rate measures the percentage of profitable trades or periods within each regime, indicating consistency of positive performance.

**Mathematical Formula:**
```
Hit Rate = (Number of profitable periods) / (Total number of periods)
```

**Regime-Specific Implementation:**
```python
def hit_rate_per_regime(returns, regime_labels, benchmark_returns=None):
    """
    Calculate hit rate for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Basic hit rate (positive returns)
        positive_returns = regime_returns > 0
        hit_rate = np.mean(positive_returns)
        
        # Hit rate vs benchmark
        if benchmark_returns is not None:
            regime_benchmark = benchmark_returns[regime_mask]
            outperformance = regime_returns > regime_benchmark
            hit_rate_vs_benchmark = np.mean(outperformance)
        else:
            hit_rate_vs_benchmark = None
        
        # Statistical significance
        n_periods = len(regime_returns)
        n_successes = np.sum(positive_returns)
        
        # Binomial test for hit rate significance
        if n_periods > 0:
            # Test against 50% null hypothesis
            expected_successes = n_periods * 0.5
            std_error = np.sqrt(n_periods * 0.5 * 0.5)
            z_score = (n_successes - expected_successes) / std_error if std_error > 0 else 0
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            p_value = 1.0
        
        metrics[regime] = {
            'hit_rate': hit_rate,
            'hit_rate_vs_benchmark': hit_rate_vs_benchmark,
            'total_periods': n_periods,
            'profitable_periods': n_successes,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'confidence_interval': calculate_hit_rate_ci(hit_rate, n_periods)
        }
    
    return metrics

def calculate_hit_rate_ci(hit_rate, n_periods, confidence=0.95):
    """Calculate confidence interval for hit rate using Wilson score"""
    if n_periods == 0:
        return (0, 0)
    
    z = norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n_periods
    center = (hit_rate + z**2 / (2 * n_periods)) / denominator
    margin = z * np.sqrt((hit_rate * (1 - hit_rate) + z**2 / (4 * n_periods)) / n_periods) / denominator
    
    return (center - margin, center + margin)
```

**Interpretation Guidelines:**
- Hit Rate > 60%: Excellent consistency
- Hit Rate 55-60%: Good consistency
- Hit Rate 50-55%: Moderate consistency
- Hit Rate < 50%: Poor consistency (more losses than gains)
- Significant p-value (<0.05) indicates hit rate is statistically different from 50%
- Wide confidence intervals indicate unreliable estimates (need more data)

### 4.2 Profit Factor per Regime

**Definition and Purpose:**
Profit factor measures the ratio of total profits to total losses within each regime, indicating overall profitability efficiency.

**Mathematical Formula:**
```
Profit Factor = Σ(positive returns) / |Σ(negative returns)|
```

**Regime-Specific Implementation:**
```python
def profit_factor_per_regime(returns, regime_labels):
    """
    Calculate profit factor for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Separate positive and negative returns
        positive_returns = regime_returns[regime_returns > 0]
        negative_returns = regime_returns[regime_returns < 0]
        
        total_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
        total_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
        
        # Profit factor calculation
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0:
            profit_factor = float('inf')  # All profits, no losses
        else:
            profit_factor = 0  # No profits, no losses
        
        # Additional metrics
        n_profitable = len(positive_returns)
        n_losing = len(negative_returns)
        total_periods = len(regime_returns)
        
        # Average win and loss
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        metrics[regime] = {
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': total_profit - total_loss,
            'n_profitable': n_profitable,
            'n_losing': n_losing,
            'total_periods': total_periods,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profitability_classification': classify_profit_factor(profit_factor)
        }
    
    return metrics

def classify_profit_factor(profit_factor):
    """Classify profit factor into performance categories"""
    if profit_factor >= 2.0:
        return "excellent"
    elif profit_factor >= 1.5:
        return "very_good"
    elif profit_factor >= 1.25:
        return "good"
    elif profit_factor >= 1.0:
        return "breakeven"
    elif profit_factor >= 0.75:
        return "poor"
    else:
        return "very_poor"
```

**Interpretation Guidelines:**
- Profit Factor > 2.0: Excellent profitability
- Profit Factor 1.5-2.0: Very good profitability
- Profit Factor 1.25-1.5: Good profitability
- Profit Factor 1.0-1.25: Marginal profitability
- Profit Factor < 1.0: Unprofitable regime
- Values > 3.0 may indicate overfitting or exceptional market conditions
- Consider sample size when interpreting (need sufficient observations)

### 4.3 Average Win/Loss Ratio per Regime

**Definition and Purpose:**
The win/loss ratio compares the average magnitude of profitable trades to losing trades, providing insight into the size vs. frequency tradeoff.

**Mathematical Formula:**
```
Win/Loss Ratio = Average Profit / |Average Loss|
```

**Regime-Specific Implementation:**
```python
def win_loss_ratio_per_regime(returns, regime_labels):
    """
    Calculate win/loss ratio for each regime
    """
    metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        # Separate positive and negative returns
        positive_returns = regime_returns[regime_returns > 0]
        negative_returns = regime_returns[regime_returns < 0]
        
        # Calculate averages
        if len(positive_returns) > 0:
            avg_win = np.mean(positive_returns)
            median_win = np.median(positive_returns)
            max_win = np.max(positive_returns)
            min_win = np.min(positive_returns)
        else:
            avg_win = median_win = max_win = min_win = 0
        
        if len(negative_returns) > 0:
            avg_loss = abs(np.mean(negative_returns))
            median_loss = abs(np.median(negative_returns))
            max_loss = abs(np.min(negative_returns))  # Most negative
            min_loss = abs(np.max(negative_returns))  # Least negative
        else:
            avg_loss = median_loss = max_loss = min_loss = 0
        
        # Win/Loss ratio
        if avg_loss > 0:
            win_loss_ratio = avg_win / avg_loss
            median_win_loss_ratio = median_win / median_loss
        else:
            win_loss_ratio = float('inf') if avg_win > 0 else 0
            median_win_loss_ratio = float('inf') if median_win > 0 else 0
        
        # Additional statistics
        n_profitable = len(positive_returns)
        n_losing = len(negative_returns)
        total_periods = len(regime_returns)
        
        metrics[regime] = {
            'win_loss_ratio': win_loss_ratio,
            'median_win_loss_ratio': median_win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'median_win': median_win,
            'median_loss': median_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'min_win': min_win,
            'min_loss': min_loss,
            'n_profitable': n_profitable,
            'n_losing': n_losing,
            'win_rate': n_profitable / total_periods if total_periods > 0 else 0,
            'expectancy': (win_loss_ratio * (n_profitable / total_periods) - 
                          (n_losing / total_periods)) if total_periods > 0 else 0,
            'risk_reward_profile': classify_risk_reward(win_loss_ratio, n_profitable / total_periods if total_periods > 0 else 0)
        }
    
    return metrics

def classify_risk_reward(win_loss_ratio, win_rate):
    """Classify the risk-reward profile of the regime"""
    if win_loss_ratio > 2.0 and win_rate > 0.5:
        return "excellent_risk_reward"
    elif win_loss_ratio > 1.5 and win_rate > 0.45:
        return "very_good_risk_reward"
    elif win_loss_ratio > 1.2 and win_rate > 0.4:
        return "good_risk_reward"
    elif win_loss_ratio > 1.0:
        return "moderate_risk_reward"
    elif win_rate > 0.6:
        return "high_frequency_low_magnitude"
    else:
        return "poor_risk_reward"
```

**Interpretation Guidelines:**
- Win/Loss Ratio > 2.0: Excellent risk-reward (wins twice as large as losses)
- Win/Loss Ratio 1.5-2.0: Very good risk-reward
- Win/Loss Ratio 1.2-1.5: Good risk-reward
- Win/Loss Ratio 1.0-1.2: Marginal risk-reward
- Win/Loss Ratio < 1.0: Poor risk-reward (losses larger than wins)
- Consider in conjunction with win rate (expectancy = (Win/Loss × Win Rate) - Loss Rate)
- High ratios with low win rates may indicate infrequent but large wins

## Regime-Specific Implementation Considerations

### 5.1 Data Requirements and Sample Size

**Minimum Sample Size Guidelines:**
- **Basic metrics** (mean, std): Minimum 30 observations per regime
- **Distribution analysis** (skewness, kurtosis): Minimum 100 observations per regime
- **Risk metrics** (VaR, ES): Minimum 200 observations per regime
- **Performance persistence**: Minimum 50 trades/periods per regime

**Data Quality Considerations:**
- Ensure sufficient observations within each regime for statistical significance
- Address regime transition periods (may need exclusion buffers)
- Consider time-varying nature of regimes (non-stationarity)
- Account for survivorship bias in historical data

### 5.2 Regime Transition Handling

**Transition Period Effects:**
- Implement buffer periods around regime transitions
- Use rolling windows to smooth regime classifications
- Consider transition regimes as separate categories
- Apply weighting schemes to reduce transition impact

**Implementation Example:**
```python
def handle_regime_transitions(regime_labels, buffer_size=5):
    """
    Handle regime transitions with buffer periods
    """
    smoothed_labels = regime_labels.copy()
    
    # Identify transition points
    transitions = np.where(regime_labels[:-1] != regime_labels[1:])[0]
    
    # Apply buffer around transitions
    for transition in transitions:
        start = max(0, transition - buffer_size)
        end = min(len(regime_labels), transition + buffer_size + 1)
        
        # Mark as transition regime
        smoothed_labels[start:end] = -1  # Special transition label
    
    return smoothed_labels
```

### 5.3 Temporal Considerations

**Time-of-Day Effects:**
- Account for intraday patterns in regime analysis
- Consider separate regimes for different trading sessions
- Adjust metrics for market hours and holidays

**Seasonal Adjustments:**
- Implement seasonal decomposition for regime metrics
- Consider monthly/quarterly regime patterns
- Adjust for calendar effects (earnings, holidays)

### 5.4 Multi-Asset Considerations

**Cross-Asset Regime Analysis:**
- Calculate metrics for asset classes within regimes
- Consider correlation structures across regimes
- Implement regime-specific portfolio optimization

**Implementation Example:**
```python
def multi_asset_regime_metrics(returns_dict, regime_labels):
    """
    Calculate regime metrics for multiple assets
    """
    assets = list(returns_dict.keys())
    regime_metrics = {}
    
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_metrics[regime] = {}
        
        # Individual asset metrics
        for asset in assets:
            asset_returns = returns_dict[asset][regime_mask]
            regime_metrics[regime][asset] = calculate_basic_metrics(asset_returns)
        
        # Portfolio metrics
        portfolio_returns = np.mean([returns_dict[asset][regime_mask] for asset in assets], axis=0)
        regime_metrics[regime]['portfolio'] = calculate_basic_metrics(portfolio_returns)
        
        # Correlation matrix
        regime_data = np.column_stack([returns_dict[asset][regime_mask] for asset in assets])
        regime_metrics[regime]['correlation_matrix'] = np.corrcoef(regime_data.T)
    
    return regime_metrics
```

## Interpretation Guidelines

### 6.1 Metric Integration Framework

**Composite Regime Score:**
```python
def calculate_regime_economic_score(metrics_dict, weights=None):
    """
    Calculate composite economic score for each regime
    """
    if weights is None:
        weights = {
            'return_metrics': 0.3,
            'risk_metrics': 0.25,
            'statistical_measures': 0.2,
            'performance_persistence': 0.25
        }
    
    regime_scores = {}
    
    for regime in metrics_dict.keys():
        score_components = {}
        
        # Return metrics score
        return_score = normalize_return_metrics(metrics_dict[regime]['returns'])
        score_components['return_metrics'] = return_score
        
        # Risk metrics score
        risk_score = normalize_risk_metrics(metrics_dict[regime]['risk'])
        score_components['risk_metrics'] = risk_score
        
        # Statistical measures score
        stats_score = normalize_statistical_metrics(metrics_dict[regime]['statistics'])
        score_components['statistical_measures'] = stats_score
        
        # Performance persistence score
        persistence_score = normalize_persistence_metrics(metrics_dict[regime]['persistence'])
        score_components['performance_persistence'] = persistence_score
        
        # Weighted composite score
        composite_score = sum(weights[component] * score_components[component] 
                            for component in weights.keys())
        
        regime_scores[regime] = {
            'composite_score': composite_score,
            'component_scores': score_components,
            'regime_classification': classify_regime_economic_value(composite_score)
        }
    
    return regime_scores

def classify_regime_economic_value(score):
    """Classify regime based on economic value"""
    if score > 0.8:
        return "highly_valuable"
    elif score > 0.6:
        return "valuable"
    elif score > 0.4:
        return "moderately_valuable"
    elif score > 0.2:
        return "minimally_valuable"
    else:
        return "not_valuable"
```

### 6.2 Benchmarking and Comparison

**Relative Performance Assessment:**
- Compare regime metrics against appropriate benchmarks
- Use peer group comparisons for context
- Implement statistical significance testing
- Consider economic significance vs. statistical significance

**Benchmark Implementation:**
```python
def benchmark_regime_metrics(regime_metrics, benchmark_metrics):
    """
    Compare regime metrics against benchmarks
    """
    benchmark_comparison = {}
    
    for regime in regime_metrics.keys():
        comparison = {}
        
        for metric_category in regime_metrics[regime].keys():
            if metric_category in benchmark_metrics:
                regime_values = regime_metrics[regime][metric_category]
                benchmark_values = benchmark_metrics[metric_category]
                
                comparison[metric_category] = {
                    'relative_performance': calculate_relative_performance(regime_values, benchmark_values),
                    'statistical_significance': test_significance(regime_values, benchmark_values),
                    'economic_significance': assess_economic_significance(regime_values, benchmark_values)
                }
        
        benchmark_comparison[regime] = comparison
    
    return benchmark_comparison
```

### 6.3 Decision Rules and Thresholds

**Regime Selection Criteria:**
```python
def evaluate_regime_tradeability(regime_metrics, thresholds=None):
    """
    Evaluate whether a regime is suitable for trading
    """
    if thresholds is None:
        thresholds = {
            'min_sharpe': 0.5,
            'max_drawdown': -0.15,
            'min_hit_rate': 0.52,
            'min_profit_factor': 1.1,
            'max_volatility': 0.4,
            'min_observations': 100
        }
    
    tradeable_regimes = {}
    
    for regime in regime_metrics.keys():
        metrics = regime_metrics[regime]
        
        # Check threshold conditions
        conditions = {
            'sharpe_acceptable': metrics['risk']['sharpe_ratio'] >= thresholds['min_sharpe'],
            'drawdown_acceptable': metrics['risk']['max_drawdown'] >= thresholds['max_drawdown'],
            'hit_rate_acceptable': metrics['persistence']['hit_rate'] >= thresholds['min_hit_rate'],
            'profit_factor_acceptable': metrics['persistence']['profit_factor'] >= thresholds['min_profit_factor'],
            'volatility_acceptable': metrics['risk']['realized_volatility'] <= thresholds['max_volatility'],
            'sufficient_data': metrics['metadata']['observations'] >= thresholds['min_observations']
        }
        
        # Overall tradeability score
        tradeability_score = sum(conditions.values()) / len(conditions)
        
        tradeable_regimes[regime] = {
            'is_tradeable': tradeability_score >= 0.7,  # 70% of conditions met
            'tradeability_score': tradeability_score,
            'conditions_met': conditions,
            'recommendation': generate_regime_recommendation(conditions, tradeability_score)
        }
    
    return tradeable_regimes

def generate_regime_recommendation(conditions, score):
    """Generate trading recommendation for regime"""
    if score >= 0.9:
        return "highly_recommended"
    elif score >= 0.7:
        return "recommended"
    elif score >= 0.5:
        return "conditional"
    else:
        return "not_recommended"
```

## Implementation Best Practices

### 7.1 Code Structure and Organization

**Modular Design:**
```python
class RegimeEconomicAnalyzer:
    """
    Comprehensive regime economic analysis system
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.return_calculator = ReturnMetricsCalculator()
        self.risk_calculator = RiskMetricsCalculator()
        self.stats_calculator = StatisticalMeasuresCalculator()
        self.persistence_calculator = PerformancePersistenceCalculator()
    
    def analyze_regimes(self, returns, regime_labels, economic_data=None):
        """
        Perform comprehensive regime economic analysis
        """
        # Validate inputs
        self._validate_inputs(returns, regime_labels)
        
        # Calculate all metrics
        analysis_results = {
            'return_metrics': self.return_calculator.calculate_all(returns, regime_labels),
            'risk_metrics': self.risk_calculator.calculate_all(returns, regime_labels),
            'statistical_measures': self.stats_calculator.calculate_all(returns, regime_labels, economic_data),
            'performance_persistence': self.persistence_calculator.calculate_all(returns, regime_labels)
        }
        
        # Add composite scores and recommendations
        analysis_results['economic_scores'] = self._calculate_economic_scores(analysis_results)
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        
        return analysis_results
    
    def _validate_inputs(self, returns, regime_labels):
        """Validate input data"""
        if len(returns) != len(regime_labels):
            raise ValueError("Returns and regime labels must have same length")
        
        if len(np.unique(regime_labels)) < 2:
            raise ValueError("Need at least 2 regimes for meaningful analysis")
    
    def _default_config(self):
        """Default configuration for analysis"""
        return {
            'min_observations_per_regime': 30,
            'confidence_levels': [0.01, 0.05, 0.1],
            'risk_free_rate': 0.02,
            'benchmark_returns': None
        }
```

### 7.2 Performance Optimization

**Efficient Calculations:**
```python
class OptimizedRegimeAnalyzer:
    """
    Performance-optimized regime analysis
    """
    
    def __init__(self):
        self._cache = {}
    
    def calculate_metrics_vectorized(self, returns, regime_labels):
        """
        Vectorized calculation for better performance
        """
        unique_regimes = np.unique(regime_labels)
        results = {}
        
        # Pre-compute common values
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_returns = returns[mask]
            regime_drawdown = drawdown[mask]
            
            # Vectorized calculations
            results[regime] = {
                'mean_return': np.mean(regime_returns),
                'volatility': np.std(regime_returns, ddof=1),
                'sharpe_ratio': self._calculate_sharpe_vectorized(regime_returns),
                'max_drawdown': np.min(regime_drawdown),
                'hit_rate': np.mean(regime_returns > 0)
            }
        
        return results
    
    def _calculate_sharpe_vectorized(self, returns):
        """Vectorized Sharpe ratio calculation"""
        excess_returns = returns - 0.02/252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
```

### 7.3 Robustness and Validation

**Statistical Validation:**
```python
class RegimeMetricsValidator:
    """
    Validate regime metrics for robustness
    """
    
    def validate_metrics(self, metrics_dict, returns, regime_labels):
        """
        Perform comprehensive validation of regime metrics
        """
        validation_results = {}
        
        for regime in metrics_dict.keys():
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            validation_results[regime] = {
                'sample_size_adequate': self._check_sample_size(regime_returns),
                'normality_test': self._test_normality(regime_returns),
                'stationarity_test': self._test_stationarity(regime_returns),
                'outlier_analysis': self._analyze_outliers(regime_returns),
                'bootstrap_validation': self._bootstrap_validation(regime_returns, metrics_dict[regime])
            }
        
        return validation_results
    
    def _bootstrap_validation(self, returns, metrics, n_bootstrap=1000):
        """
        Bootstrap validation for metric stability
        """
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            sample_metrics = self._calculate_basic_metrics(sample)
            bootstrap_metrics.append(sample_metrics)
        
        # Calculate confidence intervals
        bootstrap_array = np.array(bootstrap_metrics)
        confidence_intervals = {}
        
        for metric in bootstrap_metrics[0].keys():
            values = bootstrap_array[:, list(bootstrap_metrics[0].keys()).index(metric)]
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return confidence_intervals
```

## Code Examples and Integration

### 8.1 Complete Implementation Example

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import warnings

class ComprehensiveRegimeEconomicAnalyzer:
    """
    Complete implementation of economic relevance metrics for regime analysis
    """
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self._validate_config()
    
    def analyze_regimes(self, returns, regime_labels, economic_data=None, benchmark_returns=None):
        """
        Perform comprehensive economic analysis of market regimes
        
        Args:
            returns: Array or Series of asset returns
            regime_labels: Array of regime classifications
            economic_data: DataFrame of economic variables (optional)
            benchmark_returns: Array of benchmark returns (optional)
        
        Returns:
            Dictionary containing comprehensive regime analysis
        """
        # Convert to numpy arrays if needed
        returns = np.asarray(returns)
        regime_labels = np.asarray(regime_labels)
        
        # Validate inputs
        self._validate_inputs(returns, regime_labels)
        
        # Calculate all metric categories
        analysis_results = {
            'metadata': self._calculate_metadata(regime_labels),
            'return_metrics': self._calculate_return_metrics(returns, regime_labels),
            'risk_metrics': self._calculate_risk_metrics(returns, regime_labels),
            'statistical_measures': self._calculate_statistical_measures(
                returns, regime_labels, economic_data),
            'performance_persistence': self._calculate_performance_persistence(
                returns, regime_labels, benchmark_returns)
        }
        
        # Add composite analysis
        analysis_results['economic_scores'] = self._calculate_economic_scores(analysis_results)
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        analysis_results['validation'] = self._validate_analysis(analysis_results, returns, regime_labels)
        
        return analysis_results
    
    def _calculate_metadata(self, regime_labels):
        """Calculate metadata about regime distribution"""
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        total_observations = len(regime_labels)
        
        return {
            'total_observations': total_observations,
            'n_regimes': len(unique_regimes),
            'regime_counts': dict(zip(unique_regimes, counts)),
            'regime_proportions': dict(zip(unique_regimes, counts / total_observations)),
            'regime_labels': unique_regimes.tolist()
        }
    
    def _calculate_return_metrics(self, returns, regime_labels):
        """Calculate all return-based metrics"""
        return {
            'average_returns': self.average_return_per_regime(returns, regime_labels),
            'return_distribution': self.return_distribution_metrics(returns, regime_labels),
            'risk_adjusted_returns': self.risk_adjusted_returns(returns, regime_labels, self.config['risk_free_rate'])
        }
    
    def _calculate_risk_metrics(self, returns, regime_labels):
        """Calculate all risk metrics"""
        return {
            'sharpe_ratio': self.sharpe_ratio_per_regime(returns, regime_labels, self.config['risk_free_rate']),
            'sortino_ratio': self.sortino_ratio_per_regime(returns, regime_labels, self.config['risk_free_rate']),
            'volatility': self.volatility_per_regime(returns, regime_labels),
            'max_drawdown': self.max_drawdown_per_regime(returns, regime_labels),
            'calmar_ratio': self.calmar_ratio_per_regime(returns, regime_labels),
            'var': self.var_per_regime(returns, regime_labels, self.config['confidence_levels']),
            'expected_shortfall': self.expected_shortfall_per_regime(returns, regime_labels, self.config['confidence_levels'])
        }
    
    def _calculate_statistical_measures(self, returns, regime_labels, economic_data):
        """Calculate all statistical measures"""
        return {
            'cv_per_regime': self.cv_per_regime(returns, regime_labels),
            'within_regime_cv': self.within_regime_cv(economic_data, regime_labels, self.config['economic_variables']) if economic_data is not None else {},
            'between_regime_cv': self.between_regime_cv(economic_data, regime_labels, self.config['economic_variables']) if economic_data is not None else {},
            'skewness_kurtosis': self.skewness_kurtosis_per_regime(returns, regime_labels)
        }
    
    def _calculate_performance_persistence(self, returns, regime_labels, benchmark_returns):
        """Calculate all performance persistence metrics"""
        return {
            'hit_rate': self.hit_rate_per_regime(returns, regime_labels, benchmark_returns),
            'profit_factor': self.profit_factor_per_regime(returns, regime_labels),
            'win_loss_ratio': self.win_loss_ratio_per_regime(returns, regime_labels)
        }
    
    # Include all the metric calculation methods from previous sections
    # (average_return_per_regime, sharpe_ratio_per_regime, etc.)
    
    def _get_default_config(self):
        """Default configuration for analysis"""
        return {
            'risk_free_rate': 0.02,
            'confidence_levels': [0.01, 0.05, 0.1],
            'economic_variables': [
                'interest_rate', 'inflation_rate', 'gdp_growth', 'unemployment_rate',
                'volatility_index', 'credit_spread', 'term_spread', 'market_sentiment'
            ],
            'min_observations_per_regime': 30,
            'bootstrap_samples': 1000,
            'significance_level': 0.05
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['risk_free_rate', 'confidence_levels', 'min_observations_per_regime']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")
    
    def _validate_inputs(self, returns, regime_labels):
        """Validate input data"""
        if len(returns) != len(regime_labels):
            raise ValueError("Returns and regime labels must have same length")
        
        if len(np.unique(regime_labels)) < 2:
            raise ValueError("Need at least 2 regimes for meaningful analysis")
        
        # Check minimum observations per regime
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        for regime, count in zip(unique_regimes, counts):
            if count < self.config['min_observations_per_regime']:
                warnings.warn(f"Regime {regime} has only {count} observations, which is below the recommended minimum of {self.config['min_observations_per_regime']}")
```

### 8.2 Integration with Trading Systems

```python
class RegimeAwareTradingSystem:
    """
    Integration example for regime-aware trading system
    """
    
    def __init__(self, regime_analyzer, trading_strategy):
        self.regime_analyzer = regime_analyzer
        self.trading_strategy = trading_strategy
        self.current_regime = None
        self.regime_performance_history = []
    
    def update_regime_analysis(self, returns, regime_labels, economic_data=None):
        """
        Update regime analysis and adjust trading strategy
        """
        # Perform comprehensive regime analysis
        analysis_results = self.regime_analyzer.analyze_regimes(
            returns, regime_labels, economic_data)
        
        # Identify current most recent regime
        self.current_regime = regime_labels[-1]
        
        # Get regime-specific recommendations
        recommendations = analysis_results['recommendations'][self.current_regime]
        
        # Adjust trading strategy based on regime characteristics
        self._adjust_strategy_for_regime(analysis_results['return_metrics'][self.current_regime],
                                        analysis_results['risk_metrics'][self.current_regime],
                                        recommendations)
        
        # Store performance history
        self.regime_performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime': self.current_regime,
            'analysis': analysis_results,
            'recommendations': recommendations
        })
        
        return analysis_results
    
    def _adjust_strategy_for_regime(self, return_metrics, risk_metrics, recommendations):
        """
        Adjust trading strategy parameters based on regime characteristics
        """
        # Example strategy adjustments based on regime metrics
        if recommendations['is_tradeable']:
            # Adjust position sizing based on volatility
            volatility_adjustment = min(1.0, 0.2 / risk_metrics['volatility']['realized_volatility'])
            
            # Adjust risk limits based on drawdown
            drawdown_adjustment = max(0.5, 1.0 + risk_metrics['max_drawdown']['max_drawdown'])
            
            # Combine adjustments
            overall_adjustment = volatility_adjustment * drawdown_adjustment
            
            # Apply to trading strategy
            self.trading_strategy.adjust_position_sizing(overall_adjustment)
            self.trading_strategy.set_risk_limits(
                max_drawdown=risk_metrics['max_drawdown']['max_drawdown'] * 0.8)
        else:
            # Reduce exposure for non-tradeable regimes
            self.trading_strategy.reduce_exposure(0.5)
    
    def get_regime_summary(self):
        """
        Get summary of regime performance history
        """
        if not self.regime_performance_history:
            return "No regime history available"
        
        recent_analysis = self.regime_performance_history[-1]['analysis']
        
        summary = {
            'current_regime': self.current_regime,
            'regime_economic_score': recent_analysis['economic_scores'][self.current_regime]['composite_score'],
            'key_metrics': {
                'sharpe_ratio': recent_analysis['risk_metrics']['sharpe_ratio'][self.current_regime],
                'max_drawdown': recent_analysis['risk_metrics']['max_drawdown'][self.current_regime]['max_drawdown'],
                'hit_rate': recent_analysis['performance_persistence']['hit_rate'][self.current_regime]['hit_rate'],
                'profit_factor': recent_analysis['performance_persistence']['profit_factor'][self.current_regime]['profit_factor']
            },
            'recommendation': recent_analysis['recommendations'][self.current_regime]['recommendation']
        }
        
        return summary
```

This comprehensive documentation provides a complete framework for implementing economic relevance metrics in regime ensemble training systems. The code examples and implementation guidelines can be directly used for practical deployment in trading and investment systems.

<update_todo_list>
<todos>
[x] Research return-based metrics for regime analysis
[x] Document risk metrics with formulas and implementation details
[x] Research statistical measures for regime evaluation
[x] Document performance persistence metrics
[x] Create comprehensive technical documentation with implementation guidelines
[x] Add regime-specific calculation considerations
[x] Include interpretation guidelines for all metrics
</todos>
</update_todo_list>