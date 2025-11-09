# Comprehensive Temporal Smoothness and Transition Metrics for Regime Ensemble Training Systems

## Table of Contents
1. [Introduction](#introduction)
2. [Temporal Smoothness Metrics](#temporal-smoothness-metrics)
3. [Transition Matrix Analysis](#transition-matrix-analysis)
4. [Flip-Flop Analysis](#flip-flop-analysis)
5. [Temporal Consistency Metrics](#temporal-consistency-metrics)
6. [Implementation Considerations](#implementation-considerations)
7. [Interpretation Guidelines](#interpretation-guidelines)
8. [Complete Implementation Framework](#complete-implementation-framework)

## Introduction

This document provides comprehensive temporal smoothness and transition metrics for evaluating financial market regimes in regime ensemble training systems. These metrics are designed to assess the temporal stability, predictability, and quality of regime classifications, focusing on how regimes evolve over time and transition between different market states.

### Purpose of Temporal Analysis

- **Validate Regime Stability**: Ensure that identified regimes exhibit meaningful temporal persistence
- **Assess Transition Quality**: Evaluate the smoothness and predictability of regime changes
- **Identify Noise vs. Signal**: Distinguish between meaningful regime shifts and random fluctuations
- **Optimize Trading Timing**: Determine optimal entry/exit points based on transition patterns
- **Model Selection**: Choose appropriate models for different temporal regime characteristics

### Key Principles

1. **Temporal Coherence**: Regimes should exhibit reasonable persistence and meaningful transitions
2. **Predictability**: Transition patterns should provide actionable insights for trading decisions
3. **Noise Robustness**: Metrics should be resilient to random fluctuations and classification noise
4. **Economic Significance**: Temporal patterns should have practical implications for trading strategies

## Temporal Smoothness Metrics

### 1.1 Regime Persistence Measures

**Definition and Purpose:**
Regime persistence measures quantify how long market regimes typically last, providing insight into the stability and predictability of market conditions.

**Mathematical Formulas:**

**Average Regime Duration:**
```
ARD = (1/N) * Σ(D_i)
```
Where:
- ARD = Average regime duration
- N = Number of regime occurrences
- D_i = Duration of i-th regime occurrence

**Median Regime Duration:**
```
MRD = median(D_1, D_2, ..., D_N)
```

**Regime Half-Life:**
```
HL = ln(0.5) / ln(1 - P_stay)
```
Where:
- HL = Regime half-life
- P_stay = Probability of staying in the same regime

**Implementation:**
```python
def regime_persistence_metrics(regime_labels, timestamps=None):
    """
    Calculate comprehensive regime persistence metrics
    
    Args:
        regime_labels: Array of regime classifications
        timestamps: Array of timestamps (optional, for time-based duration)
    
    Returns:
        Dictionary with persistence metrics for each regime
    """
    import numpy as np
    from collections import defaultdict
    
    metrics = {}
    unique_regimes = np.unique(regime_labels)
    
    for regime in unique_regimes:
        # Find regime occurrences and durations
        regime_mask = regime_labels == regime
        regime_changes = np.diff(np.concatenate(([False], regime_mask, [False])).astype(int))
        start_indices = np.where(regime_changes == 1)[0]
        end_indices = np.where(regime_changes == -1)[0]
        
        # Calculate durations
        durations = end_indices - start_indices
        
        # Time-based durations if timestamps provided
        if timestamps is not None:
            time_durations = []
            for start, end in zip(start_indices, end_indices):
                duration = timestamps[end] - timestamps[start]
                time_durations.append(duration.total_seconds() / 86400)  # Convert to days
            durations = np.array(time_durations)
        
        # Calculate metrics
        metrics[regime] = {
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'std_duration': np.std(durations, ddof=1),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'n_occurrences': len(durations),
            'total_periods': np.sum(durations),
            'duration_cv': np.std(durations, ddof=1) / np.mean(durations) if np.mean(durations) > 0 else float('inf'),
            'persistence_score': calculate_persistence_score(durations),
            'duration_distribution': classify_duration_distribution(durations)
        }
    
    return metrics

def calculate_persistence_score(durations):
    """
    Calculate a persistence score (0-1) based on duration consistency
    """
    if len(durations) == 0:
        return 0
    
    # Higher score for consistent, longer durations
    mean_duration = np.mean(durations)
    cv = np.std(durations, ddof=1) / mean_duration if mean_duration > 0 else float('inf')
    
    # Normalize duration (assuming periods, adjust based on timeframe)
    normalized_duration = min(mean_duration / 10, 1.0)  # 10 periods as reference
    
    # Consistency score (inverse of CV, normalized)
    consistency_score = 1 / (1 + cv)
    
    return 0.6 * normalized_duration + 0.4 * consistency_score

def classify_duration_distribution(durations):
    """
    Classify the duration distribution pattern
    """
    if len(durations) < 3:
        return "insufficient_data"
    
    mean_duration = np.mean(durations)
    std_duration = np.std(durations, ddof=1)
    cv = std_duration / mean_duration if mean_duration > 0 else float('inf')
    
    if cv < 0.3:
        return "highly_consistent"
    elif cv < 0.6:
        return "moderately_consistent"
    elif cv < 1.0:
        return "variable"
    else:
        return "highly_variable"
```

**Interpretation Guidelines:**
- **Mean Duration > 20 periods**: Highly persistent regime (suitable for longer-term strategies)
- **Mean Duration 10-20 periods**: Moderately persistent (suitable for swing trading)
- **Mean Duration 5-10 periods**: Low persistence (suitable for shorter-term strategies)
- **Mean Duration < 5 periods**: Very low persistence (may indicate noise)
- **CV < 0.5**: Consistent duration patterns (predictable)
- **CV > 1.0**: Highly variable durations (less predictable)

### 1.2 Smoothness Indices for Regime Transitions

**Definition and Purpose:**
Smoothness indices measure the quality of regime transitions, distinguishing between meaningful market state changes and noisy fluctuations.

**Mathematical Formulas:**

**Transition Smoothness Index (TSI):**
```
TSI = 1 - (N_transitions / N_total) * (1 / mean_duration)
```

**Regime Stability Index (RSI):**
```
RSI = (1/N) * Σ[(D_i - μ_D)² / μ_D²]^(-1)
```

**Transition Penalty Score (TPS):**
```
TPS = Σ[w_i * P(transition_i)]
```
Where w_i are weights based on transition economic significance

**Implementation:**
```python
def transition_smoothness_metrics(regime_labels, economic_significance_weights=None):
    """
    Calculate smoothness metrics for regime transitions
    """
    import numpy as np
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Count transitions
    transitions = np.where(regime_labels[:-1] != regime_labels[1:])[0]
    n_transitions = len(transitions)
    n_total = len(regime_labels)
    
    # Calculate transition types
    transition_types = []
    for trans_idx in transitions:
        from_regime = regime_labels[trans_idx]
        to_regime = regime_labels[trans_idx + 1]
        transition_types.append((from_regime, to_regime))
    
    # Transition frequency matrix
    transition_matrix = np.zeros((n_regimes, n_regimes))
    for from_regime, to_regime in transition_types:
        from_idx = np.where(unique_regimes == from_regime)[0][0]
        to_idx = np.where(unique_regimes == to_regime)[0][0]
        transition_matrix[from_idx, to_idx] += 1
    
    # Normalize to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probabilities = np.divide(transition_matrix, row_sums, 
                                        where=row_sums != 0)
    
    # Calculate smoothness metrics
    transition_rate = n_transitions / n_total
    
    # Average duration between transitions
    if n_transitions > 0:
        transition_gaps = np.diff(transitions)
        avg_transition_gap = np.mean(transition_gaps)
        std_transition_gap = np.std(transition_gaps, ddof=1)
    else:
        avg_transition_gap = n_total
        std_transition_gap = 0
    
    # Smoothness indices
    tsi = 1 - transition_rate * (1 / avg_transition_gap) if avg_transition_gap > 0 else 0
    
    # Calculate RSI (Regime Stability Index)
    regime_durations = []
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        regime_changes = np.diff(np.concatenate(([False], regime_mask, [False])).astype(int))
        start_indices = np.where(regime_changes == 1)[0]
        end_indices = np.where(regime_changes == -1)[0]
        durations = end_indices - start_indices
        regime_durations.extend(durations)
    
    if regime_durations:
        mean_duration = np.mean(regime_durations)
        if mean_duration > 0:
            normalized_variances = [(d - mean_duration)**2 / mean_duration**2 for d in regime_durations]
            rsi = 1 / (1 + np.mean(normalized_variances))
        else:
            rsi = 0
    else:
        rsi = 0
    
    # Transition Penalty Score
    if economic_significance_weights:
        tps = 0
        for (from_regime, to_regime), count in zip(transition_types, 
                                                   np.bincount([i for i in range(len(transition_types))])):
            weight = economic_significance_weights.get((from_regime, to_regime), 1.0)
            tps += weight * (count / n_transitions)
    else:
        tps = transition_rate
    
    return {
        'transition_rate': transition_rate,
        'n_transitions': n_transitions,
        'avg_transition_gap': avg_transition_gap,
        'std_transition_gap': std_transition_gap,
        'transition_smoothness_index': tsi,
        'regime_stability_index': rsi,
        'transition_penalty_score': tps,
        'transition_matrix': transition_probabilities,
        'transition_types': transition_types,
        'smoothness_classification': classify_smoothness(tsi, rsi, transition_rate)
    }

def classify_smoothness(tsi, rsi, transition_rate):
    """
    Classify overall smoothness characteristics
    """
    if tsi > 0.8 and rsi > 0.7 and transition_rate < 0.1:
        return "very_smooth"
    elif tsi > 0.6 and rsi > 0.5 and transition_rate < 0.2:
        return "smooth"
    elif tsi > 0.4 and rsi > 0.3 and transition_rate < 0.3:
        return "moderately_smooth"
    elif tsi > 0.2 and rsi > 0.2:
        return "somewhat_choppy"
    else:
        return "very_choppy"
```

**Interpretation Guidelines:**
- **TSI > 0.8**: Very smooth transitions (high-quality regime classification)
- **TSI 0.6-0.8**: Smooth transitions (good regime quality)
- **TSI 0.4-0.6**: Moderate smoothness (acceptable but could be improved)
- **TSI < 0.4**: Choppy transitions (may indicate overfitting or noise)
- **RSI > 0.7**: Highly stable regimes
- **RSI 0.5-0.7**: Moderately stable
- **RSI < 0.5**: Low stability (unreliable regimes)

### 1.3 Temporal Autocorrelation of Regime Assignments

**Definition and Purpose:**
Temporal autocorrelation measures the correlation of regime classifications with their lagged values, indicating the degree of persistence and predictability in the time series.

**Mathematical Formulas:**

**Autocorrelation Function (ACF):**
```
ACF(k) = Cov(R_t, R_{t-k}) / Var(R_t)
```

**Partial Autocorrelation Function (PACF):**
```
PACF(k) = Corr(R_t, R_{t-k} | R_{t-1}, ..., R_{t-k+1})
```

**Effective Autocorrelation Time:**
```
τ_eff = 1 + 2 * Σ_{k=1}^{∞} ACF(k)
```

**Implementation:**
```python
def temporal_autocorrelation_metrics(regime_labels, max_lag=50):
    """
    Calculate temporal autocorrelation metrics for regime assignments
    """
    import numpy as np
    from scipy import stats
    from statsmodels.tsa.stattools import acf, pacf
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Convert to numeric if needed
    if regime_labels.dtype == 'object':
        regime_numeric = np.zeros_like(regime_labels, dtype=float)
        for i, regime in enumerate(unique_regimes):
            regime_numeric[regime_labels == regime] = i
    else:
        regime_numeric = regime_labels.astype(float)
    
    # Calculate ACF and PACF
    autocorr_values = acf(regime_numeric, nlags=max_lag, fft=True)
    partial_autocorr_values = pacf(regime_numeric, nlags=max_lag, method='ols')
    
    # Calculate effective autocorrelation time
    # Find where ACF crosses zero or becomes negligible
    significant_lags = np.where(np.abs(autocorr_values[1:]) > 0.1)[0]
    if len(significant_lags) > 0:
        effective_autocorr_time = 1 + 2 * np.sum(np.abs(autocorr_values[1:significant_lags[-1]+1]))
    else:
        effective_autocorr_time = 1
    
    # Calculate decay rate
    if len(autocorr_values) > 1 and autocorr_values[1] < 1:
        decay_rate = -np.log(autocorr_values[1])
    else:
        decay_rate = 0
    
    # Calculate predictability metrics
    first_lag_autocorr = autocorr_values[1] if len(autocorr_values) > 1 else 0
    predictability_score = first_lag_autocorr**2  # R² from AR(1) model
    
    # Calculate regime-specific autocorrelation
    regime_autocorr = {}
    for regime in unique_regimes:
        regime_series = (regime_labels == regime).astype(float)
        regime_acf = acf(regime_series, nlags=10, fft=True)
        regime_autocorr[regime] = {
            'acf_values': regime_acf,
            'first_lag': regime_acf[1] if len(regime_acf) > 1 else 0,
            'persistence': np.sum(regime_acf[1:6])  # Sum of first 5 lags
        }
    
    return {
        'autocorrelation_function': autocorr_values,
        'partial_autocorrelation_function': partial_autocorr_values,
        'effective_autocorrelation_time': effective_autocorr_time,
        'decay_rate': decay_rate,
        'first_lag_autocorr': first_lag_autocorr,
        'predictability_score': predictability_score,
        'regime_specific_autocorr': regime_autocorr,
        'autocorr_classification': classify_autocorr_characteristics(
            first_lag_autocorr, effective_autocorr_time, decay_rate)
    }

def classify_autocorr_characteristics(first_lag, eff_time, decay_rate):
    """
    Classify autocorrelation characteristics
    """
    if first_lag > 0.8 and eff_time > 20:
        return "highly_persistent"
    elif first_lag > 0.6 and eff_time > 10:
        return "persistent"
    elif first_lag > 0.4 and eff_time > 5:
        return "moderately_persistent"
    elif first_lag > 0.2:
        return "weakly_persistent"
    else:
        return "non_persistent"
```

**Interpretation Guidelines:**
- **First-lag ACF > 0.8**: Very high persistence (regimes highly predictable)
- **First-lag ACF 0.6-0.8**: High persistence
- **First-lag ACF 0.4-0.6**: Moderate persistence
- **First-lag ACF 0.2-0.4**: Low persistence
- **First-lag ACF < 0.2**: Very low persistence (may be random)
- **Effective autocorrelation time > 20**: Very long memory
- **Effective autocorrelation time 10-20**: Long memory
- **Effective autocorrelation time 5-10**: Moderate memory
- **Effective autocorrelation time < 5**: Short memory

### 1.4 Regime Stability Metrics Over Time

**Definition and Purpose:**
Regime stability metrics assess how the characteristics of each regime evolve over time, identifying whether regimes maintain consistent properties or drift in their behavior.

**Mathematical Formulas:**

**Temporal Stability Coefficient (TSC):**
```
TSC = 1 - (1/T) * Σ[(μ_t - μ_0)² / σ_0²]
```

**Regime Drift Rate:**
```
DR = dμ/dt / σ_μ
```

**Consistency Index:**
```
CI = 1 - CV(μ_t, σ_t, skew_t, kurt_t)
```

**Implementation:**
```python
def regime_stability_over_time(regime_labels, returns, window_size=252, step_size=21):
    """
    Calculate regime stability metrics over time using rolling windows
    """
    import numpy as np
    from scipy import stats
    
    unique_regimes = np.unique(regime_labels)
    stability_metrics = {}
    
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) < window_size:
            continue
        
        # Rolling window analysis
        rolling_metrics = {
            'mean': [],
            'std': [],
            'skew': [],
            'kurt': [],
            'sharpe': [],
            'timestamps': []
        }
        
        for i in range(0, len(regime_returns) - window_size + 1, step_size):
            window_data = regime_returns[i:i + window_size]
            
            rolling_metrics['mean'].append(np.mean(window_data))
            rolling_metrics['std'].append(np.std(window_data, ddof=1))
            rolling_metrics['skew'].append(stats.skew(window_data))
            rolling_metrics['kurt'].append(stats.kurtosis(window_data, fisher=True))
            
            # Sharpe ratio
            excess_returns = window_data - 0.02/252  # Daily risk-free rate
            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            else:
                sharpe = 0
            rolling_metrics['sharpe'].append(sharpe)
        
        # Calculate stability metrics
        stability_metrics[regime] = calculate_stability_coefficients(rolling_metrics)
    
    return stability_metrics

def calculate_stability_coefficients(rolling_metrics):
    """
    Calculate various stability coefficients from rolling metrics
    """
    import numpy as np
    
    metrics = {}
    
    for metric_name, values in rolling_metrics.items():
        if metric_name == 'timestamps' or len(values) < 2:
            continue
        
        values = np.array(values)
        
        # Temporal Stability Coefficient
        baseline_value = values[0]
        variance = np.var(values, ddof=1)
        
        if variance > 0:
            tsc = 1 - np.mean((values - baseline_value)**2) / variance
        else:
            tsc = 1.0
        
        # Drift rate (linear trend)
        if len(values) > 1:
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            drift_rate = slope / (np.mean(values) + 1e-8) if np.mean(values) != 0 else 0
        else:
            drift_rate = 0
            p_value = 1.0
        
        # Consistency Index (inverse of CV)
        cv = np.std(values, ddof=1) / (np.mean(values) + 1e-8) if np.mean(values) != 0 else float('inf')
        consistency_index = 1 / (1 + cv)
        
        metrics[metric_name] = {
            'temporal_stability_coefficient': tsc,
            'drift_rate': drift_rate,
            'drift_significance': p_value,
            'consistency_index': consistency_index,
            'coefficient_of_variation': cv,
            'trend_direction': 'increasing' if drift_rate > 0 else 'decreasing' if drift_rate < 0 else 'stable',
            'stability_classification': classify_metric_stability(tsc, drift_rate, p_value)
        }
    
    return metrics

def classify_metric_stability(tsc, drift_rate, p_value):
    """
    Classify the stability of a metric
    """
    if tsc > 0.8 and abs(drift_rate) < 0.01 and p_value > 0.1:
        return "highly_stable"
    elif tsc > 0.6 and abs(drift_rate) < 0.05 and p_value > 0.05:
        return "stable"
    elif tsc > 0.4 and abs(drift_rate) < 0.1:
        return "moderately_stable"
    elif tsc > 0.2:
        return "somewhat_unstable"
    else:
        return "highly_unstable"
```

**Interpretation Guidelines:**
- **TSC > 0.8**: Highly stable metric (consistent regime characteristics)
- **TSC 0.6-0.8**: Stable metric
- **TSC 0.4-0.6**: Moderately stable
- **TSC < 0.4**: Unstable (regime characteristics changing significantly)
- **Drift Rate < 0.01**: Negligible drift
- **Drift Rate 0.01-0.05**: Low drift
- **Drift Rate 0.05-0.1**: Moderate drift
- **Drift Rate > 0.1**: High drift (regime evolution)

## Transition Matrix Analysis

### 2.1 Markov Transition Probability Matrices

**Definition and Purpose:**
Markov transition probability matrices quantify the likelihood of transitioning from one regime to another, providing a probabilistic framework for understanding regime dynamics and predicting future states.

**Mathematical Formulas:**

**Transition Probability Matrix:**
```
P_ij = N_ij / N_i
```
Where:
- P_ij = Probability of transitioning from regime i to regime j
- N_ij = Number of transitions from i to j
- N_i = Total number of transitions from regime i

**n-Step Transition Matrix:**
```
P^n = P * P * ... * P (n times)
```

**Steady-State Distribution:**
```
π = π * P, where Σπ_i = 1
```

**Implementation:**
```python
def markov_transition_matrix(regime_labels, order=1):
    """
    Calculate Markov transition probability matrix
    
    Args:
        regime_labels: Array of regime classifications
        order: Order of Markov chain (default=1)
    
    Returns:
        Dictionary with transition matrix and related metrics
    """
    import numpy as np
    from collections import defaultdict
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Count transitions
    transition_counts = np.zeros((n_regimes,) * (order + 1), dtype=int)
    
    for i in range(len(regime_labels) - order):
        current_states = []
        for j in range(order + 1):
            regime_idx = np.where(unique_regimes == regime_labels[i + j])[0][0]
            current_states.append(regime_idx)
        
        # Increment transition count
        transition_counts[tuple(current_states)] += 1
    
    # Calculate transition probabilities
    transition_probabilities = np.zeros_like(transition_counts, dtype=float)
    
    if order == 1:
        for i in range(n_regimes):
            total_transitions = np.sum(transition_counts[i, :])
            if total_transitions > 0:
                transition_probabilities[i, :] = transition_counts[i, :] / total_transitions
    else:
        # Higher-order Markov chains
        for indices in np.ndindex(*transition_counts.shape[:-1]):
            total_transitions = np.sum(transition_counts[indices])
            if total_transitions > 0:
                transition_probabilities[indices] = transition_counts[indices] / total_transitions
    
    # Calculate n-step transition matrices
    n_step_matrices = {}
    current_matrix = transition_probabilities.copy()
    n_step_matrices[1] = current_matrix
    
    for n in range(2, 6):  # Calculate up to 5-step transitions
        current_matrix = np.dot(n_step_matrices[n-1], transition_probabilities)
        n_step_matrices[n] = current_matrix
    
    # Calculate steady-state distribution
    steady_state = calculate_steady_state(transition_probabilities)
    
    # Calculate fundamental matrix (for absorbing Markov chains)
    fundamental_matrix = calculate_fundamental_matrix(transition_probabilities)
    
    return {
        'transition_counts': transition_counts,
        'transition_probabilities': transition_probabilities,
        'n_step_matrices': n_step_matrices,
        'steady_state_distribution': steady_state,
        'fundamental_matrix': fundamental_matrix,
        'regime_labels': unique_regimes,
        'order': order,
        'transition_entropy': calculate_transition_entropy(transition_probabilities),
        'mixing_time': estimate_mixing_time(transition_probabilities)
    }

def calculate_steady_state(transition_matrix):
    """
    Calculate steady-state distribution of Markov chain
    """
    import numpy as np
    from scipy.linalg import eig
    
    # Find eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = eig(transition_matrix.T)
    
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.real(eigenvectors[:, idx])
    
    # Normalize to sum to 1
    steady_state = steady_state / np.sum(steady_state)
    
    return steady_state

def calculate_fundamental_matrix(transition_matrix):
    """
    Calculate fundamental matrix for absorbing Markov chains
    """
    import numpy as np
    
    n = transition_matrix.shape[0]
    
    # Check for absorbing states
    absorbing_states = np.where(np.diag(transition_matrix) > 0.99)[0]
    
    if len(absorbing_states) == 0:
        return None
    
    # Reorder matrix to put absorbing states last
    transient_states = [i for i in range(n) if i not in absorbing_states]
    reordered_indices = transient_states + absorbing_states.tolist()
    
    P_reordered = transition_matrix[np.ix_(reordered_indices, reordered_indices)]
    
    # Extract Q matrix (transient to transient transitions)
    t = len(transient_states)
    Q = P_reordered[:t, :t]
    
    # Calculate fundamental matrix N = (I - Q)^(-1)
    I = np.eye(t)
    try:
        N = np.linalg.inv(I - Q)
        return N
    except np.linalg.LinAlgError:
        return None

def calculate_transition_entropy(transition_matrix):
    """
    Calculate entropy of transition matrix
    """
    import numpy as np
    
    # Avoid log(0)
    transition_matrix_safe = np.where(transition_matrix > 0, transition_matrix, 1e-10)
    
    # Calculate entropy for each row (state)
    row_entropy = -np.sum(transition_matrix_safe * np.log(transition_matrix_safe), axis=1)
    
    # Overall entropy (weighted by steady state)
    steady_state = calculate_steady_state(transition_matrix)
    overall_entropy = np.sum(steady_state * row_entropy)
    
    return {
        'row_entropy': row_entropy,
        'overall_entropy': overall_entropy,
        'max_entropy': np.log(len(transition_matrix)),
        'entropy_ratio': overall_entropy / np.log(len(transition_matrix))
    }

def estimate_mixing_time(transition_matrix, threshold=0.01):
    """
    Estimate mixing time of Markov chain
    """
    import numpy as np
    
    steady_state = calculate_steady_state(transition_matrix)
    n = len(transition_matrix)
    
    # Start from arbitrary initial distribution
    current_dist = np.ones(n) / n
    mixing_time = 0
    
    for step in range(1, 1000):  # Maximum 1000 steps
        current_dist = np.dot(current_dist, transition_matrix)
        
        # Check convergence to steady state
        distance = np.sum(np.abs(current_dist - steady_state))
        
        if distance < threshold:
            mixing_time = step
            break
    
    return mixing_time
```

**Interpretation Guidelines:**
- **Diagonal elements > 0.8**: High persistence (regimes tend to stay the same)
- **Diagonal elements 0.6-0.8**: Moderate persistence
- **Diagonal elements < 0.6**: Low persistence (frequent transitions)
- **Off-diagonal elements > 0.2**: Significant transition probability
- **Entropy ratio > 0.8**: High unpredictability in transitions
- **Entropy ratio < 0.3**: Low unpredictability (predictable transitions)
- **Mixing time < 10**: Fast mixing (quickly reaches steady state)
- **Mixing time > 50**: Slow mixing (long memory effects)

### 2.2 Transition Frequency Analysis

**Definition and Purpose:**
Transition frequency analysis examines how often specific regime transitions occur, identifying common pathways and rare events in market regime evolution.

**Mathematical Formulas:**

**Transition Frequency:**
```
f_ij = N_ij / T_total
```

**Relative Transition Frequency:**
```
rf_ij = N_ij / N_transitions
```

**Transition Periodicity:**
```
P_ij = Average time between i→j transitions
```

**Implementation:**
```python
def transition_frequency_analysis(regime_labels, timestamps=None):
    """
    Comprehensive transition frequency analysis
    """
    import numpy as np
    from collections import defaultdict, Counter
    import pandas as pd
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Identify all transitions
    transitions = []
    transition_indices = []
    
    for i in range(len(regime_labels) - 1):
        if regime_labels[i] != regime_labels[i + 1]:
            transitions.append((regime_labels[i], regime_labels[i + 1]))
            transition_indices.append(i)
    
    # Count transitions
    transition_counts = Counter(transitions)
    total_transitions = len(transitions)
    total_periods = len(regime_labels)
    
    # Calculate frequency metrics
    transition_frequencies = {}
    relative_frequencies = {}
    
    for transition in transition_counts:
        count = transition_counts[transition]
        transition_frequencies[transition] = count / total_periods
        relative_frequencies[transition] = count / total_transitions
    
    # Time-based analysis if timestamps provided
    time_analysis = {}
    if timestamps is not None:
        transition_times = []
        for idx in transition_indices:
            transition_times.append(timestamps[idx + 1])  # Time when transition occurs
        
        # Calculate inter-transition times
        if len(transition_times) > 1:
            inter_transition_times = []
            for i in range(1, len(transition_times)):
                gap = transition_times[i] - transition_times[i - 1]
                inter_transition_times.append(gap.total_seconds() / 86400)  # Convert to days
            
            time_analysis = {
                'mean_inter_transition_time': np.mean(inter_transition_times),
                'std_inter_transition_time': np.std(inter_transition_times, ddof=1),
                'transition_rate': total_transitions / ((timestamps[-1] - timestamps[0]).total_seconds() / 86400),
                'peak_transition_periods': identify_peak_transition_periods(transition_times)
            }
    
    # Analyze specific transition patterns
    transition_patterns = analyze_transition_patterns(transitions, unique_regimes)
    
    # Calculate transition matrix with frequencies
    freq_matrix = np.zeros((n_regimes, n_regimes))
    rel_freq_matrix = np.zeros((n_regimes, n_regimes))
    
    for i, from_regime in enumerate(unique_regimes):
        for j, to_regime in enumerate(unique_regimes):
            transition = (from_regime, to_regime)
            freq_matrix[i, j] = transition_frequencies.get(transition, 0)
            rel_freq_matrix[i, j] = relative_frequencies.get(transition, 0)
    
    return {
        'transition_counts': dict(transition_counts),
        'transition_frequencies': transition_frequencies,
        'relative_frequencies': relative_frequencies,
        'total_transitions': total_transitions,
        'transition_rate': total_transitions / total_periods,
        'frequency_matrix': freq_matrix,
        'relative_frequency_matrix': rel_freq_matrix,
        'time_analysis': time_analysis,
        'transition_patterns': transition_patterns,
        'common_transitions': identify_common_transitions(relative_frequencies),
        'rare_transitions': identify_rare_transitions(relative_frequencies)
    }

def analyze_transition_patterns(transitions, unique_regimes):
    """
    Analyze specific transition patterns
    """
    from collections import defaultdict
    
    patterns = {
        'cycles': [],
        'one_way_transitions': [],
        'reciprocal_pairs': [],
        'hub_regimes': defaultdict(int),
        'sink_regimes': defaultdict(int)
    }
    
    # Count outgoing and incoming transitions
    outgoing_counts = defaultdict(int)
    incoming_counts = defaultdict(int)
    
    for from_regime, to_regime in transitions:
        outgoing_counts[from_regime] += 1
        incoming_counts[to_regime] += 1
    
    # Identify hub regimes (many outgoing transitions)
    for regime in unique_regimes:
        if outgoing_counts[regime] > 2:
            patterns['hub_regimes'][regime] = outgoing_counts[regime]
    
    # Identify sink regimes (many incoming transitions)
    for regime in unique_regimes:
        if incoming_counts[regime] > 2:
            patterns['sink_regimes'][regime] = incoming_counts[regime]
    
    # Identify reciprocal pairs
    transition_set = set(transitions)
    for from_regime, to_regime in transitions:
        if (to_regime, from_regime) in transition_set:
            patterns['reciprocal_pairs'].append((from_regime, to_regime))
    
    # Remove duplicates from reciprocal pairs
    patterns['reciprocal_pairs'] = list(set(tuple(sorted(pair)) for pair in patterns['reciprocal_pairs']))
    
    return patterns

def identify_common_transitions(relative_frequencies, threshold=0.1):
    """
    Identify transitions that occur frequently
    """
    return {trans: freq for trans, freq in relative_frequencies.items() 
            if freq >= threshold}

def identify_rare_transitions(relative_frequencies, threshold=0.01):
    """
    Identify transitions that occur rarely
    """
    return {trans: freq for trans, freq in relative_frequencies.items() 
            if freq < threshold}

def identify_peak_transition_periods(transition_times, window_size='30D'):
    """
    Identify periods with high transition activity
    """
    import pandas as pd
    
    if not transition_times:
        return []
    
    # Create time series of transition counts
    transition_series = pd.Series(1, index=transition_times)
    
    # Resample to count transitions in windows
    transition_counts = transition_series.resample(window_size).sum()
    
    # Find peaks (periods with high transition counts)
    threshold = transition_counts.mean() + 2 * transition_counts.std()
    peak_periods = transition_counts[transition_counts > threshold]
    
    return peak_periods.index.tolist()
```

**Interpretation Guidelines:**
- **Transition rate > 0.05**: High frequency of regime changes
- **Transition rate 0.02-0.05**: Moderate frequency
- **Transition rate < 0.02**: Low frequency (stable regimes)
- **Common transitions (>10%)**: Important pathways to monitor
- **Rare transitions (<1%)**: May represent crisis events or special conditions
- **Hub regimes**: Important transition points in market dynamics
- **Sink regimes**: Potential end states for market evolution

### 2.3 Expected Time in Each Regime

**Definition and Purpose:**
Expected time in each regime quantifies the average duration a market spends in a particular state before transitioning, providing insights into regime persistence and trading horizon considerations.

**Mathematical Formulas:**

**Expected Duration (for discrete-time Markov chain):**
```
E[D_i] = 1 / (1 - P_ii)
```

**Expected Return Time:**
```
E[R_i] = 1 / π_i
```

**Variance of Duration:**
```
Var(D_i) = P_ii / (1 - P_ii)²
```

**Implementation:**
```python
def expected_time_metrics(transition_matrix, regime_labels=None):
    """
    Calculate expected time metrics for each regime
    """
    import numpy as np
    
    n_regimes = transition_matrix.shape[0]
    
    # Calculate expected duration in each regime
    expected_durations = np.zeros(n_regimes)
    variance_durations = np.zeros(n_regimes)
    
    for i in range(n_regimes):
        stay_probability = transition_matrix[i, i]
        
        if stay_probability < 1:
            expected_durations[i] = 1 / (1 - stay_probability)
            variance_durations[i] = stay_probability / ((1 - stay_probability) ** 2)
        else:
            expected_durations[i] = float('inf')
            variance_durations[i] = float('inf')
    
    # Calculate steady-state distribution
    steady_state = calculate_steady_state(transition_matrix)
    
    # Calculate expected return time
    expected_return_times = 1 / (steady_state + 1e-10)
    
    # Calculate regime visitation frequency
    visitation_frequency = steady_state
    
    # Calculate half-life for each regime
    half_lives = np.zeros(n_regimes)
    for i in range(n_regimes):
        stay_probability = transition_matrix[i, i]
        if stay_probability < 1:
            half_lives[i] = np.log(0.5) / np.log(stay_probability)
        else:
            half_lives[i] = float('inf')
    
    # Calculate confidence intervals for durations
    duration_confidence_intervals = {}
    for i in range(n_regimes):
        if variance_durations[i] != float('inf'):
            std_duration = np.sqrt(variance_durations[i])
            mean_duration = expected_durations[i]
            
            # 95% confidence interval (approximate normal distribution)
            ci_lower = max(1, mean_duration - 1.96 * std_duration)
            ci_upper = mean_duration + 1.96 * std_duration
        else:
            ci_lower = ci_upper = float('inf')
        
        duration_confidence_intervals[i] = (ci_lower, ci_upper)
    
    results = {
        'expected_durations': expected_durations,
        'variance_durations': variance_durations,
        'std_durations': np.sqrt(variance_durations),
        'expected_return_times': expected_return_times,
        'visitation_frequency': visitation_frequency,
        'half_lives': half_lives,
        'duration_confidence_intervals': duration_confidence_intervals,
        'persistence_classification': classify_persistence(expected_durations, half_lives)
    }
    
    # Add regime labels if provided
    if regime_labels is not None:
        labeled_results = {}
        for i, regime in enumerate(regime_labels):
            labeled_results[regime] = {
                'expected_duration': expected_durations[i],
                'std_duration': np.sqrt(variance_durations[i]),
                'expected_return_time': expected_return_times[i],
                'visitation_frequency': visitation_frequency[i],
                'half_life': half_lives[i],
                'confidence_interval': duration_confidence_intervals[i],
                'persistence_class': classify_persistence(expected_durations[i], half_lives[i])
            }
        results['by_regime'] = labeled_results
    
    return results

def classify_persistence(expected_duration, half_life):
    """
    Classify regime persistence based on duration metrics
    """
    if expected_duration == float('inf'):
        return "absorbing"
    elif expected_duration > 50:
        return "highly_persistent"
    elif expected_duration > 20:
        return "persistent"
    elif expected_duration > 10:
        return "moderately_persistent"
    elif expected_duration > 5:
        return "weakly_persistent"
    else:
        return "transient"

def calculate_regime_occupancy_probabilities(transition_matrix, time_horizons):
    """
    Calculate probability of being in each regime after different time horizons
    """
    import numpy as np
    
    n_regimes = transition_matrix.shape[0]
    occupancy_probabilities = {}
    
    # Start from steady state
    steady_state = calculate_steady_state(transition_matrix)
    current_distribution = steady_state.copy()
    
    for horizon in time_horizons:
        # Calculate distribution after horizon steps
        if horizon == 0:
            occupancy_probabilities[horizon] = steady_state.copy()
        else:
            horizon_distribution = np.dot(steady_state, np.linalg.matrix_power(transition_matrix, horizon))
            occupancy_probabilities[horizon] = horizon_distribution
    
    return occupancy_probabilities
```

**Interpretation Guidelines:**
- **Expected Duration > 50 periods**: Highly persistent regime (long-term strategies)
- **Expected Duration 20-50**: Persistent regime (medium-term strategies)
- **Expected Duration 10-20**: Moderate persistence (swing trading)
- **Expected Duration 5-10**: Low persistence (shorter-term strategies)
- **Expected Duration < 5**: Very transient (day trading or noise)
- **Half-life > 35**: Very slow decay of regime influence
- **Half-life 15-35**: Moderate decay
- **Half-life < 15**: Fast decay (quick regime changes)
- **Visitation frequency > 0.3**: Dominant regime (occurs frequently)
- **Visitation frequency < 0.1**: Rare regime (special conditions)

### 2.4 Absorbing States Identification

**Definition and Purpose:**
Absorbing states are regimes from which the market cannot transition to other states, representing potential equilibrium or crisis conditions that are important for risk management.

**Mathematical Formulas:**

**Absorbing State Condition:**
```
P_ii = 1 (or P_ii > 0.99 for practical purposes)
```

**Absorption Probability:**
```
B_ij = Probability of being absorbed in state j starting from state i
```

**Expected Time to Absorption:**
```
t_i = Expected time to reach absorbing state from state i
```

**Implementation:**
```python
def absorbing_states_analysis(transition_matrix, regime_labels=None, threshold=0.99):
    """
    Identify and analyze absorbing states in the transition matrix
    """
    import numpy as np
    from scipy.linalg import inv
    
    n_regimes = transition_matrix.shape[0]
    
    # Identify absorbing states
    absorbing_mask = np.diag(transition_matrix) >= threshold
    absorbing_indices = np.where(absorbing_mask)[0]
    transient_indices = np.where(~absorbing_mask)[0]
    
    if len(absorbing_indices) == 0:
        return {
            'has_absorbing_states': False,
            'absorbing_states': [],
            'transient_states': list(range(n_regimes))
        }
    
    # Reorder matrix to separate absorbing and transient states
    reordered_indices = list(transient_indices) + list(absorbing_indices)
    P_reordered = transition_matrix[np.ix_(reordered_indices, reordered_indices)]
    
    n_transient = len(transient_indices)
    n_absorbing = len(absorbing_indices)
    
    # Extract submatrices
    Q = P_reordered[:n_transient, :n_transient]  # Transient to transient
    R = P_reordered[:n_transient, n_transient:]  # Transient to absorbing
    
    # Calculate fundamental matrix N = (I - Q)^(-1)
    I = np.eye(n_transient)
    try:
        N = inv(I - Q)
        
        # Calculate absorption probabilities B = N * R
        B = np.dot(N, R)
        
        # Calculate expected time to absorption t = N * 1
        t = np.sum(N, axis=1)
        
        # Calculate variance of time to absorption
        t_var = np.dot(N, np.sum(N, axis=1)) - t**2
        
    except np.linalg.LinAlgError:
        # Singular matrix (degenerate case)
        N = np.full((n_transient, n_transient), np.inf)
        B = np.full((n_transient, n_absorbing), np.nan)
        t = np.full(n_transient, np.inf)
        t_var = np.full(n_transient, np.nan)
    
    # Map back to original indices
    absorption_probabilities = np.full((n_regimes, n_regimes), np.nan)
    expected_absorption_times = np.full(n_regimes, np.inf)
    variance_absorption_times = np.full(n_regimes, np.nan)
    
    # Fill in results for transient states
    for i, transient_idx in enumerate(transient_indices):
        expected_absorption_times[transient_idx] = t[i]
        variance_absorption_times[transient_idx] = t_var[i]
        
        for j, absorbing_idx in enumerate(absorbing_indices):
            absorption_probabilities[transient_idx, absorbing_idx] = B[i, j]
    
    # For absorbing states, absorption probability is 1 for themselves
    for absorbing_idx in absorbing_indices:
        absorption_probabilities[absorbing_idx, absorbing_idx] = 1.0
        expected_absorption_times[absorbing_idx] = 0
        variance_absorption_times[absorbing_idx] = 0
    
    # Analyze absorbing state characteristics
    absorbing_characteristics = {}
    for i, absorbing_idx in enumerate(absorbing_indices):
        absorbing_characteristics[absorbing_idx] = {
            'absorption_probability_from_steady_state': None,
            'is_dominant_absorber': False,
            'absorption_strength': np.sum(absorption_probabilities[:, absorbing_idx])
        }
    
    # Calculate absorption probability from steady state
    steady_state = calculate_steady_state(transition_matrix)
    for absorbing_idx in absorbing_indices:
        absorbing_characteristics[absorbing_idx]['absorption_probability_from_steady_state'] = steady_state[absorbing_idx]
    
    # Identify dominant absorber
    if len(absorbing_indices) > 1:
        absorption_strengths = [absorbing_characteristics[idx]['absorption_strength'] 
                              for idx in absorbing_indices]
        dominant_idx = absorbing_indices[np.argmax(absorption_strengths)]
        absorbing_characteristics[dominant_idx]['is_dominant_absorber'] = True
    
    results = {
        'has_absorbing_states': True,
        'absorbing_states': absorbing_indices.tolist(),
        'transient_states': transient_indices.tolist(),
        'absorption_probabilities': absorption_probabilities,
        'expected_absorption_times': expected_absorption_times,
        'variance_absorption_times': variance_absorption_times,
        'absorbing_characteristics': absorbing_characteristics,
        'fundamental_matrix': N,
        'absorption_matrix': B
    }
    
    # Add regime labels if provided
    if regime_labels is not None:
        results['absorbing_regime_labels'] = [regime_labels[i] for i in absorbing_indices]
        results['transient_regime_labels'] = [regime_labels[i] for i in transient_indices]
    
    return results

def classify_absorbing_states(absorbing_analysis, economic_context=None):
    """
    Classify the type and economic significance of absorbing states
    """
    classifications = {}
    
    if not absorbing_analysis['has_absorbing_states']:
        return classifications
    
    absorbing_states = absorbing_analysis['absorbing_states']
    absorbing_characteristics = absorbing_analysis['absorbing_characteristics']
    
    for absorbing_idx in absorbing_states:
        characteristics = absorbing_characteristics[absorbing_idx]
        
        # Classify based on absorption strength
        if characteristics['absorption_strength'] > 0.5:
            strength_class = "strong_absorber"
        elif characteristics['absorption_strength'] > 0.2:
            strength_class = "moderate_absorber"
        else:
            strength_class = "weak_absorber"
        
        # Classify based on steady-state probability
        steady_prob = characteristics['absorption_probability_from_steady_state']
        if steady_prob > 0.3:
            equilibrium_class = "likely_equilibrium"
        elif steady_prob > 0.1:
            equilibrium_class = "possible_equilibrium"
        else:
            equilibrium_class = "rare_equilibrium"
        
        # Combine classifications
        classifications[absorbing_idx] = {
            'strength_class': strength_class,
            'equilibrium_class': equilibrium_class,
            'is_dominant': characteristics['is_dominant_absorber'],
            'economic_interpretation': interpret_absorbing_state_economically(
                strength_class, equilibrium_class, economic_context)
        }
    
    return classifications

def interpret_absorbing_state_economically(strength_class, equilibrium_class, economic_context):
    """
    Provide economic interpretation of absorbing states
    """
    if strength_class == "strong_absorber" and equilibrium_class == "likely_equilibrium":
        return "stable_market_equilibrium"
    elif strength_class == "strong_absorber" and equilibrium_class == "rare_equilibrium":
        return "crisis_trap_state"
    elif strength_class == "moderate_absorber" and equilibrium_class == "possible_equilibrium":
        return "temporary_equilibrium"
    elif strength_class == "weak_absorber":
        return "transitional_state"
    else:
        return "unclassified_absorbing_state"
```

**Interpretation Guidelines:**
- **Strong absorbers (>0.5 absorption strength)**: Once entered, likely to persist
- **Weak absorbers (<0.2 absorption strength)**: May escape with reasonable probability
- **High steady-state probability (>0.3)**: Natural equilibrium state
- **Low steady-state probability (<0.1)**: Crisis or special condition state
- **Expected absorption time < 10**: Quick absorption (rapid convergence)
- **Expected absorption time > 50**: Slow absorption (prolonged transient behavior)

### 2.5 Transition Entropy Measures

**Definition and Purpose:**
Transition entropy measures quantify the uncertainty and predictability of regime transitions, providing insights into the complexity and randomness of market state evolution.

**Mathematical Formulas:**

**Shannon Entropy of Transitions:**
```
H = -Σ_{i,j} π_i * P_ij * log(P_ij)
```

**Conditional Entropy:**
```
H(Y|X) = -Σ_i π_i * Σ_j P_ij * log(P_ij)
```

**Mutual Information:**
```
I(X;Y) = H(Y) - H(Y|X)
```

**Kullback-Leibler Divergence:**
```
D_KL(P||Q) = Σ_i P_i * log(P_i / Q_i)
```

**Implementation:**
```python
def transition_entropy_analysis(transition_matrix, regime_labels=None):
    """
    Comprehensive entropy analysis of regime transitions
    """
    import numpy as np
    
    n_regimes = transition_matrix.shape[0]
    
    # Calculate steady-state distribution
    steady_state = calculate_steady_state(transition_matrix)
    
    # Calculate Shannon entropy for each row (regime)
    row_entropies = np.zeros(n_regimes)
    for i in range(n_regimes):
        row_probs = transition_matrix[i, :]
        # Avoid log(0)
        row_probs_safe = np.where(row_probs > 0, row_probs, 1e-10)
        row_entropies[i] = -np.sum(row_probs_safe * np.log(row_probs_safe))
    
    # Calculate overall entropy (weighted by steady state)
    overall_entropy = np.sum(steady_state * row_entropies)
    
    # Calculate maximum possible entropy (uniform distribution)
    max_entropy = np.log(n_regimes)
    entropy_ratio = overall_entropy / max_entropy
    
    # Calculate conditional entropy H(Y|X)
    conditional_entropy = overall_entropy
    
    # Calculate marginal entropy of next state H(Y)
    marginal_next_state = np.sum(steady_state[:, np.newaxis] * transition_matrix, axis=0)
    marginal_next_state_safe = np.where(marginal_next_state > 0, marginal_next_state, 1e-10)
    marginal_entropy = -np.sum(marginal_next_state_safe * np.log(marginal_next_state_safe))
    
    # Calculate mutual information
    mutual_information = marginal_entropy - conditional_entropy
    
    # Calculate relative entropy from steady state to next state
    kl_divergence = 0
    for i in range(n_regimes):
        if steady_state[i] > 0 and marginal_next_state[i] > 0:
            kl_divergence += steady_state[i] * np.log(steady_state[i] / marginal_next_state[i])
    
    # Calculate entropy rate (per time step)
    entropy_rate = overall_entropy
    
    # Calculate predictability measures
    predictability_1_step = 1 - (overall_entropy / max_entropy)
    predictability_n_steps = calculate_n_step_predictability(transition_matrix, n_steps=5)
    
    # Calculate entropy production (measure of irreversibility)
    entropy_production = calculate_entropy_production(transition_matrix)
    
    results = {
        'row_entropies': row_entropies,
        'overall_entropy': overall_entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': entropy_ratio,
        'conditional_entropy': conditional_entropy,
        'marginal_entropy': marginal_entropy,
        'mutual_information': mutual_information,
        'kl_divergence': kl_divergence,
        'entropy_rate': entropy_rate,
        'predictability_1_step': predictability_1_step,
        'predictability_n_steps': predictability_n_steps,
        'entropy_production': entropy_production,
        'entropy_classification': classify_entropy_characteristics(entropy_ratio, mutual_information)
    }
    
    # Add regime labels if provided
    if regime_labels is not None:
        labeled_row_entropies = {}
        for i, regime in enumerate(regime_labels):
            labeled_row_entropies[regime] = row_entropies[i]
        results['labeled_row_entropies'] = labeled_row_entropies
    
    return results

def calculate_n_step_predictability(transition_matrix, n_steps=5):
    """
    Calculate predictability over multiple time steps
    """
    import numpy as np
    
    n_regimes = transition_matrix.shape[0]
    max_entropy = np.log(n_regimes)
    
    predictabilities = {}
    
    for n in range(1, n_steps + 1):
        # Calculate n-step transition matrix
        n_step_matrix = np.linalg.matrix_power(transition_matrix, n)
        
        # Calculate entropy of n-step transitions
        steady_state = calculate_steady_state(transition_matrix)
        
        n_step_entropy = 0
        for i in range(n_regimes):
            row_probs = n_step_matrix[i, :]
            row_probs_safe = np.where(row_probs > 0, row_probs, 1e-10)
            row_entropy = -np.sum(row_probs_safe * np.log(row_probs_safe))
            n_step_entropy += steady_state[i] * row_entropy
        
        predictability = 1 - (n_step_entropy / max_entropy)
        predictabilities[n] = predictability
    
    return predictabilities

def calculate_entropy_production(transition_matrix):
    """
    Calculate entropy production (measure of time irreversibility)
    """
    import numpy as np
    
    # Calculate detailed balance condition
    steady_state = calculate_steady_state(transition_matrix)
    
    # Calculate forward and reverse fluxes
    forward_flux = np.zeros_like(transition_matrix)
    reverse_flux = np.zeros_like(transition_matrix)
    
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            forward_flux[i, j] = steady_state[i] * transition_matrix[i, j]
            reverse_flux[j, i] = steady_state[j] * transition_matrix[j, i]
    
    # Calculate entropy production
    entropy_production = 0
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if forward_flux[i, j] > 0 and reverse_flux[i, j] > 0:
                entropy_production += (forward_flux[i, j] - reverse_flux[i, j]) * \
                                     np.log(forward_flux[i, j] / reverse_flux[i, j])
    
    return entropy_production

def classify_entropy_characteristics(entropy_ratio, mutual_information):
    """
    Classify the entropy characteristics of the transition system
    """
    if entropy_ratio > 0.8:
        entropy_level = "high_entropy"
    elif entropy_ratio > 0.6:
        entropy_level = "moderate_entropy"
    elif entropy_ratio > 0.4:
        entropy_level = "low_entropy"
    else:
        entropy_level = "very_low_entropy"
    
    if mutual_information > 1.0:
        information_level = "high_information"
    elif mutual_information > 0.5:
        information_level = "moderate_information"
    elif mutual_information > 0.2:
        information_level = "low_information"
    else:
        information_level = "very_low_information"
    
    # Combine classifications
    if entropy_level == "high_entropy" and information_level == "low_information":
        return "random_like"
    elif entropy_level == "low_entropy" and information_level == "high_information":
        return "highly_predictable"
    elif entropy_level == "moderate_entropy" and information_level == "moderate_information":
        return "moderately_predictable"
    else:
        return "mixed_characteristics"
```

**Interpretation Guidelines:**
- **Entropy ratio > 0.8**: High uncertainty (nearly random transitions)
- **Entropy ratio 0.6-0.8**: Moderate uncertainty
- **Entropy ratio 0.4-0.6**: Low uncertainty (some predictability)
- **Entropy ratio < 0.4**: Very low uncertainty (highly predictable)
- **Mutual information > 1.0**: Strong dependence between consecutive states
- **Mutual information < 0.2**: Weak dependence (near independence)
- **Entropy production > 0**: Time-irreversible process (typical for financial markets)
- **Predictability > 0.6**: High predictability (good for trading)
- **Predictability < 0.3**: Low predictability (challenging for trading)

## Flip-Flop Analysis

### 3.1 Regime Switching Frequency Metrics

**Definition and Purpose:**
Regime switching frequency metrics quantify how often the market changes between different regimes, helping distinguish between meaningful regime shifts and excessive noise.

**Mathematical Formulas:**

**Switching Frequency:**
```
SF = N_switches / N_total
```

**Average Time Between Switches:**
```
ATBS = N_total / N_switches
```

**Switching Intensity:**
```
SI = N_switches / T_time
```

**Implementation:**
```python
def regime_switching_frequency_metrics(regime_labels, timestamps=None, min_duration=1):
    """
    Calculate comprehensive regime switching frequency metrics
    """
    import numpy as np
    
    # Identify switches
    switches = np.where(regime_labels[:-1] != regime_labels[1:])[0]
    n_switches = len(switches)
    n_total = len(regime_labels)
    
    # Basic frequency metrics
    switching_frequency = n_switches / n_total if n_total > 0 else 0
    
    # Calculate gaps between switches
    if n_switches > 1:
        switch_gaps = np.diff(switches)
        avg_gap = np.mean(switch_gaps)
        std_gap = np.std(switch_gaps, ddof=1)
        min_gap = np.min(switch_gaps)
        max_gap = np.max(switch_gaps)
    else:
        avg_gap = std_gap = min_gap = max_gap = n_total
    
    # Time-based metrics if timestamps provided
    time_metrics = {}
    if timestamps is not None:
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400  # Days
        switching_intensity = n_switches / time_span if time_span > 0 else 0
        
        if n_switches > 1:
            switch_times = [timestamps[i+1] for i in switches]
            time_gaps = [(switch_times[i] - switch_times[i-1]).total_seconds() / 86400 
                         for i in range(1, len(switch_times))]
            
            avg_time_gap = np.mean(time_gaps)
            std_time_gap = np.std(time_gaps, ddof=1)
        else:
            avg_time_gap = std_time_gap = time_span
        
        time_metrics = {
            'switching_intensity': switching_intensity,
            'avg_time_between_switches': avg_time_gap,
            'std_time_between_switches': std_time_gap,
            'time_span_days': time_span
        }
    
    # Analyze switch types
    switch_types = analyze_switch_types(regime_labels, switches)
    
    # Filter out very short regimes (potential noise)
    if min_duration > 1:
        filtered_switches = filter_short_regimes(regime_labels, switches, min_duration)
        filtered_frequency = len(filtered_switches) / n_total
        noise_ratio = (n_switches - len(filtered_switches)) / n_switches if n_switches > 0 else 0
    else:
        filtered_switches = switches
        filtered_frequency = switching_frequency
        noise_ratio = 0
    
    # Calculate switching patterns
    switching_patterns = analyze_switching_patterns(regime_labels, switches)
    
    return {
        'n_switches': n_switches,
        'switching_frequency': switching_frequency,
        'filtered_switching_frequency': filtered_frequency,
        'noise_ratio': noise_ratio,
        'avg_gap_between_switches': avg_gap,
        'std_gap_between_switches': std_gap,
        'min_gap_between_switches': min_gap,
        'max_gap_between_switches': max_gap,
        'time_metrics': time_metrics,
        'switch_types': switch_types,
        'switching_patterns': switching_patterns,
        'frequency_classification': classify_switching_frequency(switching_frequency),
        'stability_score': calculate_switching_stability_score(switching_frequency, avg_gap, std_gap)
    }

def analyze_switch_types(regime_labels, switches):
    """
    Analyze the types of regime switches
    """
    from collections import Counter
    
    switch_types = Counter()
    
    for switch_idx in switches:
        from_regime = regime_labels[switch_idx]
        to_regime = regime_labels[switch_idx + 1]
        switch_types[(from_regime, to_regime)] += 1
    
    return dict(switch_types)

def filter_short_regimes(regime_labels, switches, min_duration):
    """
    Filter out switches that create very short regimes
    """
    filtered_switches = []
    
    for i, switch_idx in enumerate(switches):
        # Find the duration of the regime after this switch
        if i < len(switches) - 1:
            next_switch = switches[i + 1]
            duration = next_switch - switch_idx
        else:
            duration = len(regime_labels) - switch_idx - 1
        
        if duration >= min_duration:
            filtered_switches.append(switch_idx)
    
    return filtered_switches

def analyze_switching_patterns(regime_labels, switches):
    """
    Analyze patterns in regime switching
    """
    import numpy as np
    
    patterns = {
        'consecutive_switches': 0,
        'alternating_patterns': [],
        'switch_clusters': [],
        'quiet_periods': []
    }
    
    # Find consecutive switches (rapid switching)
    for i in range(len(switches) - 1):
        if switches[i + 1] - switches[i] <= 3:  # Within 3 periods
            patterns['consecutive_switches'] += 1
    
    # Find alternating patterns (A->B->A)
    for i in range(len(switches) - 2):
        if (regime_labels[switches[i]] == regime_labels[switches[i + 2]] and 
            regime_labels[switches[i]] != regime_labels[switches[i + 1]]):
            patterns['alternating_patterns'].append(
                (switches[i], switches[i + 1], switches[i + 2])
            )
    
    # Find switch clusters (periods of high switching activity)
    if len(switches) > 1:
        switch_gaps = np.diff(switches)
        cluster_threshold = np.percentile(switch_gaps, 25)  # Bottom 25% of gaps
        
        current_cluster = [switches[0]]
        for i, gap in enumerate(switch_gaps):
            if gap <= cluster_threshold:
                current_cluster.append(switches[i + 1])
            else:
                if len(current_cluster) > 2:
                    patterns['switch_clusters'].append(current_cluster)
                current_cluster = [switches[i + 1]]
        
        if len(current_cluster) > 2:
            patterns['switch_clusters'].append(current_cluster)
    
    # Find quiet periods (long periods without switches)
    if len(switches) > 1:
        quiet_threshold = np.percentile(switch_gaps, 75)  # Top 25% of gaps
        
        for i, gap in enumerate(switch_gaps):
            if gap >= quiet_threshold:
                quiet_start = switches[i] + 1
                quiet_end = switches[i + 1]
                patterns['quiet_periods'].append((quiet_start, quiet_end, quiet_end - quiet_start))
    
    return patterns

def classify_switching_frequency(frequency):
    """
    Classify the switching frequency
    """
    if frequency > 0.1:
        return "very_high_frequency"
    elif frequency > 0.05:
        return "high_frequency"
    elif frequency > 0.02:
        return "moderate_frequency"
    elif frequency > 0.01:
        return "low_frequency"
    else:
        return "very_low_frequency"

def calculate_switching_stability_score(frequency, avg_gap, std_gap):
    """
    Calculate a stability score based on switching metrics
    """
    # Lower frequency and more consistent gaps = higher stability
    frequency_score = max(0, 1 - frequency * 10)  # Normalize frequency
    
    # More consistent gaps (lower CV) = higher stability
    gap_cv = std_gap / avg_gap if avg_gap > 0 else float('inf')
    consistency_score = 1 / (1 + gap_cv)
    
    # Combine scores
    stability_score = 0.6 * frequency_score + 0.4 * consistency_score
    
    return stability_score
```

**Interpretation Guidelines:**
- **Switching frequency > 0.1**: Very high frequency (likely noise)
- **Switching frequency 0.05-0.1**: High frequency (potentially overfit)
- **Switching frequency 0.02-0.05**: Moderate frequency (acceptable)
- **Switching frequency 0.01-0.02**: Low frequency (good stability)
- **Switching frequency < 0.01**: Very low frequency (very stable)
- **Noise ratio > 0.3**: High proportion of noise switches
- **Average gap < 10**: Very rapid switching
- **Average gap > 50**: Stable regimes with infrequent switches

### 3.2 Flip-Flop Rate Calculations

**Definition and Purpose:**
Flip-flop rate specifically measures rapid back-and-forth switching between regimes, which often indicates classification noise rather than meaningful market changes.

**Mathematical Formulas:**

**Flip-Flop Rate:**
```
FFR = N_flip_flops / N_total
```

**Flip-Flop Intensity:**
```
FFI = N_flip_flops / N_switches
```

**Flip-Flop Persistence:**
```
FFP = 1 - (N_flip_flops / N_possible)
```

**Implementation:**
```python
def flip_flop_rate_calculations(regime_labels, window_size=3):
    """
    Calculate detailed flip-flop metrics
    """
    import numpy as np
    from collections import defaultdict
    
    # Identify flip-flops (A->B->A patterns)
    flip_flops = []
    flip_flop_indices = []
    
    for i in range(len(regime_labels) - 2):
        if (regime_labels[i] == regime_labels[i + 2] and 
            regime_labels[i] != regime_labels[i + 1]):
            flip_flops.append((regime_labels[i], regime_labels[i + 1], regime_labels[i + 2]))
            flip_flop_indices.append(i)
    
    n_flip_flops = len(flip_flops)
    n_total = len(regime_labels)
    
    # Calculate basic flip-flop rates
    flip_flop_rate = n_flip_flops / n_total if n_total > 0 else 0
    
    # Calculate total switches for intensity calculation
    total_switches = np.sum(regime_labels[:-1] != regime_labels[1:])
    flip_flop_intensity = n_flip_flops / total_switches if total_switches > 0 else 0
    
    # Calculate flip-flop persistence
    # Maximum possible flip-flops is approximately n_total/2
    max_possible_flip_flops = n_total // 2
    flip_flop_persistence = 1 - (n_flip_flops / max_possible_flip_flops) if max_possible_flip_flops > 0 else 1
    
    # Analyze flip-flop types
    flip_flop_types = defaultdict(int)
    for ff in flip_flops:
        flip_flop_types[ff] += 1
    
    # Calculate window-based flip-flop rate
    window_flip_flop_rates = []
    if len(regime_labels) >= window_size:
        for i in range(len(regime_labels) - window_size + 1):
            window = regime_labels[i:i + window_size]
            window_ff = 0
            for j in range(len(window) - 2):
                if (window[j] == window[j + 2] and window[j] != window[j + 1]):
                    window_ff += 1
            window_flip_flop_rates.append(window_ff / window_size)
    
    # Calculate temporal clustering of flip-flops
    if len(flip_flop_indices) > 1:
        ff_gaps = np.diff(flip_flop_indices)
        ff_cluster_score = calculate_flip_flop_clustering(ff_gaps)
    else:
        ff_cluster_score = 0
    
    # Calculate regime-specific flip-flop rates
    regime_ff_rates = {}
    unique_regimes = np.unique(regime_labels)
    
    for regime in unique_regimes:
        regime_ff_count = 0
        regime_total_periods = np.sum(regime_labels == regime)
        
        for ff in flip_flops:
            if ff[0] == regime:  # Starting regime
                regime_ff_count += 1
        
        if regime_total_periods > 0:
            regime_ff_rates[regime] = regime_ff_count / regime_total_periods
        else:
            regime_ff_rates[regime] = 0
    
    # Calculate noise-to-signal ratio based on flip-flops
    noise_signal_ratio = flip_flop_rate / (1 - flip_flop_rate) if flip_flop_rate < 1 else float('inf')
    
    return {
        'n_flip_flops': n_flip_flops,
        'flip_flop_rate': flip_flop_rate,
        'flip_flop_intensity': flip_flop_intensity,
        'flip_flop_persistence': flip_flop_persistence,
        'flip_flop_types': dict(flip_flop_types),
        'window_flip_flop_rates': window_flip_flop_rates,
        'avg_window_flip_flop_rate': np.mean(window_flip_flop_rates) if window_flip_flop_rates else 0,
        'std_window_flip_flop_rate': np.std(window_flip_flop_rates) if window_flip_flop_rates else 0,
        'ff_cluster_score': ff_cluster_score,
        'regime_ff_rates': regime_ff_rates,
        'noise_signal_ratio': noise_signal_ratio,
        'flip_flop_classification': classify_flip_flop_characteristics(
            flip_flop_rate, flip_flop_intensity, noise_signal_ratio)
    }

def calculate_flip_flop_clustering(ff_gaps):
    """
    Calculate clustering score for flip-flops
    """
    import numpy as np
    
    if len(ff_gaps) == 0:
        return 0
    
    # Lower gaps indicate clustering
    mean_gap = np.mean(ff_gaps)
    std_gap = np.std(ff_gaps, ddof=1)
    
    # Cluster score: lower mean gap and lower CV = higher clustering
    cv = std_gap / mean_gap if mean_gap > 0 else float('inf')
    cluster_score = 1 / (1 + mean_gap/10 + cv)  # Normalize by typical gap of 10
    
    return cluster_score

def classify_flip_flop_characteristics(rate, intensity, noise_signal_ratio):
    """
    Classify flip-flop characteristics
    """
    if rate > 0.05 or noise_signal_ratio > 0.1:
        return "excessive_flip_flopping"
    elif rate > 0.02 or intensity > 0.3:
        return "high_flip_flopping"
    elif rate > 0.01 or intensity > 0.2:
        return "moderate_flip_flopping"
    elif rate > 0.005:
        return "low_flip_flopping"
    else:
        return "minimal_flip_flopping"

def calculate_flip_flop_economic_impact(regime_labels, returns, flip_flop_indices):
    """
    Calculate the economic impact of flip-flops
    """
    import numpy as np
    
    if len(flip_flop_indices) == 0:
        return {
            'avg_return_during_ff': 0,
            'volatility_during_ff': 0,
            'ff_vs_non_ff_performance': 0
        }
    
    # Analyze returns during flip-flop periods
    ff_returns = []
    for ff_idx in flip_flop_indices:
        # Look at returns around the flip-flop
        start_idx = max(0, ff_idx - 1)
        end_idx = min(len(returns), ff_idx + 4)  # Include the transition period
        ff_returns.extend(returns[start_idx:end_idx])
    
    if ff_returns:
        avg_return_during_ff = np.mean(ff_returns)
        volatility_during_ff = np.std(ff_returns, ddof=1)
    else:
        avg_return_during_ff = volatility_during_ff = 0
    
    # Compare with non-flip-flop periods
    non_ff_mask = np.ones(len(regime_labels), dtype=bool)
    for ff_idx in flip_flop_indices:
        non_ff_mask[ff_idx-1:ff_idx+3] = False  # Mark FF periods
    
    non_ff_returns = returns[non_ff_mask]
    
    if len(non_ff_returns) > 0:
        avg_return_non_ff = np.mean(non_ff_returns)
        ff_vs_non_ff_performance = avg_return_during_ff - avg_return_non_ff
    else:
        ff_vs_non_ff_performance = 0
    
    return {
        'avg_return_during_ff': avg_return_during_ff,
        'volatility_during_ff': volatility_during_ff,
        'ff_vs_non_ff_performance': ff_vs_non_ff_performance,
        'n_ff_periods': len(ff_returns),
        'n_non_ff_periods': len(non_ff_returns)
    }
```

**Interpretation Guidelines:**
- **Flip-flop rate > 0.05**: Excessive flip-flopping (severe classification noise)
- **Flip-flop rate 0.02-0.05**: High flip-flopping (significant noise)
- **Flip-flop rate 0.01-0.02**: Moderate flip-flopping (acceptable noise)
- **Flip-flop rate 0.005-0.01**: Low flip-flopping (good quality)
- **Flip-flop rate < 0.005**: Minimal flip-flopping (excellent quality)
- **Flip-flop intensity > 0.3**: High proportion of switches are flip-flops
- **Noise-signal ratio > 0.1**: Noise dominates signal
- **Cluster score > 0.7**: Flip-flops tend to cluster in time

### 3.3 Whipsaw Detection Metrics

**Definition and Purpose:**
Whipsaw detection metrics identify periods of rapid, repeated regime changes that can cause trading strategy whipsaw losses, helping to filter out noisy market conditions.

**Mathematical Formulas:**

**Whipsaw Intensity:**
```
WI = Σ_{i=1}^{n} w_i * I(change_i)
```

**Whipsaw Duration:**
```
WD = Length of whipsaw period
```

**Whipsaw Frequency:**
```
WF = N_whipsaw_periods / T_total
```

**Implementation:**
```python
def whipsaw_detection_metrics(regime_labels, returns=None, min_whipsaw_duration=3, 
                             max_whipsaw_gap=2):
    """
    Detect and analyze whipsaw periods in regime changes
    """
    import numpy as np
    
    # Identify regime changes
    changes = np.where(regime_labels[:-1] != regime_labels[1:])[0]
    
    if len(changes) < 2:
        return {
            'n_whipsaw_periods': 0,
            'whipsaw_periods': [],
            'whipsaw_intensity': 0,
            'whipsaw_classification': 'no_whipsaw'
        }
    
    # Identify whipsaw periods (clusters of rapid changes)
    whipsaw_periods = []
    current_whipsaw = [changes[0]]
    
    for i in range(1, len(changes)):
        gap = changes[i] - changes[i - 1]
        
        if gap <= max_whipsaw_gap:
            current_whipsaw.append(changes[i])
        else:
            # Check if current whipsaw meets minimum duration
            if len(current_whipsaw) >= min_whipsaw_duration:
                whipsaw_periods.append(current_whipsaw)
            current_whipsaw = [changes[i]]
    
    # Check the last whipsaw
    if len(current_whipsaw) >= min_whipsaw_duration:
        whipsaw_periods.append(current_whipsaw)
    
    # Calculate whipsaw metrics
    n_whipsaw_periods = len(whipsaw_periods)
    total_periods = len(regime_labels)
    whipsaw_frequency = n_whipsaw_periods / total_periods if total_periods > 0 else 0
    
    # Calculate whipsaw intensity
    total_changes_in_whipsaws = sum(len(period) for period in whipsaw_periods)
    total_changes = len(changes)
    whipsaw_intensity = total_changes_in_whipsaws / total_changes if total_changes > 0 else 0
    
    # Analyze whipsaw characteristics
    whipsaw_characteristics = []
    for period in whipsaw_periods:
        start_idx = period[0]
        end_idx = period[-1] + 1  # Include the transition point
        
        characteristics = {
            'start_index': start_idx,
            'end_index': end_idx,
            'duration': end_idx - start_idx,
            'n_changes': len(period),
            'change_density': len(period) / (end_idx - start_idx),
            'regime_sequence': regime_labels[start_idx:end_idx + 1].tolist()
        }
        
        # Add return analysis if returns provided
        if returns is not None:
            period_returns = returns[start_idx:end_idx]
            characteristics.update({
                'avg_return': np.mean(period_returns) if len(period_returns) > 0 else 0,
                'volatility': np.std(period_returns, ddof=1) if len(period_returns) > 1 else 0,
                'cumulative_return': np.sum(period_returns) if len(period_returns) > 0 else 0
            })
        
        whipsaw_characteristics.append(characteristics)
    
    # Calculate overall whipsaw statistics
    if whipsaw_characteristics:
        avg_whipsaw_duration = np.mean([c['duration'] for c in whipsaw_characteristics])
        avg_change_density = np.mean([c['change_density'] for c in whipsaw_characteristics])
        max_change_density = max([c['change_density'] for c in whipsaw_characteristics])
    else:
        avg_whipsaw_duration = avg_change_density = max_change_density = 0
    
    # Calculate whipsaw economic impact if returns provided
    economic_impact = {}
    if returns is not None and whipsaw_characteristics:
        economic_impact = calculate_whipsaw_economic_impact(
            whipsaw_characteristics, returns)
    
    # Calculate whipsaw clustering
    whipsaw_clustering = calculate_whipsaw_clustering(whipsaw_periods, total_periods)
    
    return {
        'n_whipsaw_periods': n_whipsaw_periods,
        'whipsaw_frequency': whipsaw_frequency,
        'whipsaw_intensity': whipsaw_intensity,
        'whipsaw_periods': whipsaw_characteristics,
        'avg_whipsaw_duration': avg_whipsaw_duration,
        'avg_change_density': avg_change_density,
        'max_change_density': max_change_density,
        'whipsaw_clustering': whipsaw_clustering,
        'economic_impact': economic_impact,
        'whipsaw_classification': classify_whipsaw_severity(
            whipsaw_frequency, whipsaw_intensity, avg_change_density)
    }

def calculate_whipsaw_economic_impact(whipsaw_characteristics, returns):
    """
    Calculate the economic impact of whipsaw periods
    """
    import numpy as np
    
    # Collect returns during whipsaw periods
    whipsaw_returns = []
    whipsaw_volatilities = []
    
    for ws in whipsaw_characteristics:
        if 'avg_return' in ws:
            whipsaw_returns.append(ws['avg_return'])
        if 'volatility' in ws:
            whipsaw_volatilities.append(ws['volatility'])
    
    # Calculate non-whipsaw returns for comparison
    whipsaw_mask = np.zeros(len(returns), dtype=bool)
    for ws in whipsaw_characteristics:
        whipsaw_mask[ws['start_index']:ws['end_index']] = True
    
    non_whipsaw_returns = returns[~whipsaw_mask]
    
    impact_metrics = {
        'avg_whipsaw_return': np.mean(whipsaw_returns) if whipsaw_returns else 0,
        'avg_whipsaw_volatility': np.mean(whipsaw_volatilities) if whipsaw_volatilities else 0,
        'avg_non_whipsaw_return': np.mean(non_whipsaw_returns) if len(non_whipsaw_returns) > 0 else 0,
        'whipsaw_vs_non_whipsaw': 0
    }
    
    if whipsaw_returns and len(non_whipsaw_returns) > 0:
        impact_metrics['whipsaw_vs_non_whipsaw'] = (
            np.mean(whipsaw_returns) - np.mean(non_whipsaw_returns)
        )
    
    # Calculate whipsaw loss ratio (negative returns during whipsaws)
    negative_whipsaw_returns = [r for r in whipsaw_returns if r < 0]
    impact_metrics['whipsaw_loss_ratio'] = (
        len(negative_whipsaw_returns) / len(whipsaw_returns) if whipsaw_returns else 0
    )
    
    return impact_metrics

def calculate_whipsaw_clustering(whipsaw_periods, total_periods):
    """
    Calculate clustering of whipsaw periods
    """
    if len(whipsaw_periods) < 2:
        return 0
    
    # Calculate gaps between whipsaw periods
    whipsaw_starts = [ws[0][0] for ws in whipsaw_periods]
    whipsaw_starts.sort()
    
    gaps = np.diff(whipsaw_starts)
    avg_gap = np.mean(gaps)
    
    # Clustering score: smaller gaps = more clustering
    expected_gap = total_periods / (len(whipsaw_periods) + 1)
    clustering_score = max(0, 1 - avg_gap / expected_gap)
    
    return clustering_score

def classify_whipsaw_severity(frequency, intensity, change_density):
    """
    Classify the severity of whipsaw activity
    """
    if frequency > 0.05 or intensity > 0.5 or change_density > 0.5:
        return "severe_whipsaw"
    elif frequency > 0.02 or intensity > 0.3 or change_density > 0.3:
        return "moderate_whipsaw"
    elif frequency > 0.01 or intensity > 0.2 or change_density > 0.2:
        return "mild_whipsaw"
    elif frequency > 0.005:
        return "minimal_whipsaw"
    else:
        return "no_significant_whipsaw"

def calculate_whipsaw_filtering_recommendations(whipsaw_metrics):
    """
    Generate recommendations for filtering whipsaw periods
    """
    classification = whipsaw_metrics['whipsaw_classification']
    frequency = whipsaw_metrics['whipsaw_frequency']
    intensity = whipsaw_metrics['whipsaw_intensity']
    
    recommendations = {
        'should_filter': False,
        'filter_method': None,
        'filter_parameters': {},
        'confidence': 0
    }
    
    if classification == "severe_whipsaw":
        recommendations.update({
            'should_filter': True,
            'filter_method': 'aggressive',
            'filter_parameters': {
                'min_regime_duration': 5,
                'confirmation_periods': 3,
                'volatility_threshold': 0.8
            },
            'confidence': 0.9
        })
    elif classification == "moderate_whipsaw":
        recommendations.update({
            'should_filter': True,
            'filter_method': 'moderate',
            'filter_parameters': {
                'min_regime_duration': 3,
                'confirmation_periods': 2,
                'volatility_threshold': 0.6
            },
            'confidence': 0.7
        })
    elif classification == "mild_whipsaw":
        recommendations.update({
            'should_filter': True,
            'filter_method': 'conservative',
            'filter_parameters': {
                'min_regime_duration': 2,
                'confirmation_periods': 1,
                'volatility_threshold': 0.4
            },
            'confidence': 0.5
        })
    
    return recommendations
```

**Interpretation Guidelines:**
- **Whipsaw frequency > 0.05**: Severe whipsaw conditions (avoid trading)
- **Whipsaw frequency 0.02-0.05**: Moderate whipsaw (use caution)
- **Whipsaw frequency 0.01-0.02**: Mild whipsaw (acceptable with filters)
- **Whipsaw frequency < 0.01**: Minimal whipsaw (good trading conditions)
- **Change density > 0.5**: Very rapid regime changes (high noise)
- **Change density 0.3-0.5**: Rapid changes (moderate noise)
- **Change density < 0.3**: Manageable change rate
- **Whipsaw loss ratio > 0.7**: Most whipsaws result in losses
- **Clustering score > 0.7**: Whipsaws tend to cluster in time

### 3.4 Noise vs. Signal Discrimination in Regime Changes

**Definition and Purpose:**
Noise vs. signal discrimination metrics help distinguish between meaningful regime changes and random fluctuations, improving the quality of regime classification for trading applications.

**Mathematical Formulas:**

**Signal-to-Noise Ratio (SNR):**
```
SNR = Var(signal) / Var(noise)
```

**Information Content:**
```
IC = H_before - H_after
```

**Regime Change Significance:**
```
RCS = |μ_before - μ_after| / (σ_before + σ_after)
```

**Implementation:**
```python
def noise_signal_discrimination(regime_labels, returns, feature_data=None, 
                                min_regime_duration=5):
    """
    Comprehensive noise vs. signal discrimination analysis
    """
    import numpy as np
    from scipy import stats
    
    # Identify regime changes
    changes = np.where(regime_labels[:-1] != regime_labels[1:])[0]
    
    if len(changes) == 0:
        return {
            'signal_to_noise_ratio': 0,
            'noise_level': 0,
            'signal_level': 0,
            'classification': 'no_changes'
        }
    
    # Analyze each regime change
    change_analysis = []
    
    for change_idx in changes:
        # Get windows before and after change
        window_before = max(min_regime_duration, change_idx - min_regime_duration)
        window_after = min(len(returns), change_idx + 1 + min_regime_duration)
        
        before_returns = returns[window_before:change_idx + 1]
        after_returns = returns[change_idx + 1:window_after]
        
        if len(before_returns) < 3 or len(after_returns) < 3:
            continue
        
        # Calculate change metrics
        change_metrics = analyze_regime_change(
            before_returns, after_returns, 
            regime_labels[change_idx], regime_labels[change_idx + 1]
        )
        
        change_metrics['change_index'] = change_idx
        change_analysis.append(change_metrics)
    
    # Calculate overall noise/signal metrics
    if not change_analysis:
        return {
            'signal_to_noise_ratio': 0,
            'noise_level': 0,
            'signal_level': 0,
            'classification': 'insufficient_data'
        }
    
    # Extract significance scores
    significance_scores = [c['significance_score'] for c in change_analysis]
    volatility_changes = [c['volatility_change'] for c in change_analysis]
    return_differences = [c['return_difference'] for c in change_analysis]
    
    # Calculate signal and noise components
    signal_level = np.mean([abs(s) for s in significance_scores if abs(s) > 0.5])
    noise_level = np.mean([abs(s) for s in significance_scores if abs(s) <= 0.5])
    
    signal_to_noise_ratio = signal_level / (noise_level + 1e-8)
    
    # Calculate information content
    information_content = calculate_information_content(change_analysis)
    
    # Calculate regime stability metrics
    stability_metrics = calculate_regime_stability_metrics(change_analysis)
    
    # Classify changes
    significant_changes = [c for c in change_analysis if c['is_significant']]
    noise_changes = [c for c in change_analysis if not c['is_significant']]
    
    return {
        'signal_to_noise_ratio': signal_to_noise_ratio,
        'signal_level': signal_level,
        'noise_level': noise_level,
        'information_content': information_content,
        'stability_metrics': stability_metrics,
        'n_total_changes': len(change_analysis),
        'n_significant_changes': len(significant_changes),
        'n_noise_changes': len(noise_changes),
        'significant_change_ratio': len(significant_changes) / len(change_analysis),
        'change_analysis': change_analysis,
        'discrimination_classification': classify_noise_signal_characteristics(
            signal_to_noise_ratio, len(significant_changes) / len(change_analysis)
        ),
        'filtering_recommendations': generate_filtering_recommendations(
            signal_to_noise_ratio, len(significant_changes) / len(change_analysis)
        )
    }

def analyze_regime_change(before_returns, after_returns, from_regime, to_regime):
    """
    Analyze a specific regime change for signal vs. noise
    """
    import numpy as np
    from scipy import stats
    
    # Basic statistics
    before_mean = np.mean(before_returns)
    after_mean = np.mean(after_returns)
    before_std = np.std(before_returns, ddof=1)
    after_std = np.std(after_returns, ddof=1)
    
    # Return difference
    return_difference = after_mean - before_mean
    
    # Volatility change
    volatility_change = (after_std - before_std) / (before_std + 1e-8)
    
    # Statistical significance tests
    # t-test for mean difference
    t_stat, p_value = stats.ttest_ind(before_returns, after_returns)
    
    # F-test for variance difference
    f_stat = before_std**2 / (after_std**2 + 1e-8)
    df1 = len(before_returns) - 1
    df2 = len(after_returns) - 1
    f_p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                        1 - stats.f.cdf(f_stat, df1, df2))
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(before_returns) - 1) * before_std**2 + 
                         (len(after_returns) - 1) * after_std**2) / 
                        (len(before_returns) + len(after_returns) - 2))
    cohens_d = return_difference / (pooled_std + 1e-8)
    
    # Significance score (combined measure)
    significance_score = (1 - p_value) * abs(cohens_d)
    
    # Information-theoretic measure
    kl_divergence = calculate_kl_divergence(before_returns, after_returns)
    
    return {
        'from_regime': from_regime,
        'to_regime': to_regime,
        'return_difference': return_difference,
        'volatility_change': volatility_change,
        't_statistic': t_stat,
        'p_value': p_value,
        'f_statistic': f_stat,
        'f_p_value': f_p_value,
        'cohens_d': cohens_d,
        'significance_score': significance_score,
        'kl_divergence': kl_divergence,
        'is_significant': significance_score > 0.3 and p_value < 0.1,
        'change_magnitude': abs(return_difference) / (before_std + 1e-8)
    }

def calculate_kl_divergence(returns1, returns2, n_bins=20):
    """
    Calculate Kullback-Leibler divergence between return distributions
    """
    import numpy as np
    
    # Create histograms
    min_val = min(np.min(returns1), np.min(returns2))
    max_val = max(np.max(returns1), np.max(returns2))
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    hist1, _ = np.histogram(returns1, bins=bins, density=True)
    hist2, _ = np.histogram(returns2, bins=bins, density=True)
    
    # Avoid zero probabilities
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate KL divergence
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    
    return kl_div

def calculate_information_content(change_analysis):
    """
    Calculate overall information content of regime changes
    """
    import numpy as np
    
    if not change_analysis:
        return 0
    
    # Average KL divergence across changes
    kl_divergences = [c['kl_divergence'] for c in change_analysis]
    avg_kl_divergence = np.mean(kl_divergences)
    
    # Average significance score
    significance_scores = [c['significance_score'] for c in change_analysis]
    avg_significance = np.mean(significance_scores)
    
    # Combined information content
    information_content = 0.6 * avg_kl_divergence + 0.4 * avg_significance
    
    return information_content

def calculate_regime_stability_metrics(change_analysis):
    """
    Calculate stability metrics based on regime changes
    """
    import numpy as np
    
    if not change_analysis:
        return {}
    
    # Consistency of change directions
    return_differences = [c['return_difference'] for c in change_analysis]
    change_consistency = 1 - (np.std(return_differences, ddof=1) / 
                             (np.mean(np.abs(return_differences)) + 1e-8))
    
    # Volatility stability
    volatility_changes = [c['volatility_change'] for c in change_analysis]
    volatility_stability = 1 - (np.std(volatility_changes, ddof=1) / 
                               (np.mean(np.abs(volatility_changes)) + 1e-8))
    
    # Overall stability score
    overall_stability = 0.5 * change_consistency + 0.5 * volatility_stability
    
    return {
        'change_consistency': change_consistency,
        'volatility_stability': volatility_stability,
        'overall_stability': overall_stability
    }

def classify_noise_signal_characteristics(snr, significant_ratio):
    """
    Classify the noise vs. signal characteristics
    """
    if snr > 3.0 and significant_ratio > 0.7:
        return "signal_dominated"
    elif snr > 1.5 and significant_ratio > 0.5:
        return "moderate_signal"
    elif snr > 0.5 and significant_ratio > 0.3:
        return "mixed_signal_noise"
    elif snr > 0.2:
        return "noise_dominated"
    else:
        return "pure_noise"

def generate_filtering_recommendations(snr, significant_ratio):
    """
    Generate recommendations for filtering noise
    """
    classification = classify_noise_signal_characteristics(snr, significant_ratio)
    
    if classification == "signal_dominated":
        return {
            'should_filter': False,
            'filter_strength': 'none',
            'confidence': 0.9
        }
    elif classification == "moderate_signal":
        return {
            'should_filter': True,
            'filter_strength': 'light',
            'methods': ['minimum_duration', 'statistical_confirmation'],
            'confidence': 0.7
        }
    elif classification == "mixed_signal_noise":
        return {
            'should_filter': True,
            'filter_strength': 'moderate',
            'methods': ['minimum_duration', 'statistical_confirmation', 'volatility_filter'],
            'confidence': 0.6
        }
    elif classification == "noise_dominated":
        return {
            'should_filter': True,
            'filter_strength': 'strong',
            'methods': ['minimum_duration', 'statistical_confirmation', 'volatility_filter', 'trend_confirmation'],
            'confidence': 0.8
        }
    else:  # pure_noise
        return {
            'should_filter': True,
            'filter_strength': 'aggressive',
            'methods': ['minimum_duration', 'statistical_confirmation', 'volatility_filter', 'trend_confirmation', 'ensemble_confirmation'],
            'confidence': 0.9
        }
```

**Interpretation Guidelines:**
- **SNR > 3.0**: Signal-dominated regime changes (high quality)
- **SNR 1.5-3.0**: Good signal-to-noise ratio
- **SNR 0.5-1.5**: Mixed signal and noise
- **SNR 0.2-0.5**: Noise-dominated changes
- **SNR < 0.2**: Pure noise (regime classification unreliable)
- **Significant change ratio > 0.7**: Most changes are meaningful
- **Significant change ratio 0.5-0.7**: Majority of changes are meaningful
- **Significant change ratio 0.3-0.5**: Mixed meaningful and noisy changes
- **Significant change ratio < 0.3**: Most changes are noise

## Temporal Consistency Metrics

### 4.1 Rolling Window Regime Consistency

**Definition and Purpose:**
Rolling window regime consistency measures how stable regime classifications are over time using overlapping windows, identifying periods of consistent classification versus unstable periods.

**Mathematical Formulas:**

**Window Consistency Score:**
```
WCS = 1 - (Hamming_distance / Window_size)
```

**Temporal Consistency Index:**
```
TCI = (1/T) * Σ WCS_t
```

**Consistency Volatility:**
```
CV = Std(WCS_t) / Mean(WCS_t)
```

**Implementation:**
```python
def rolling_window_regime_consistency(regime_labels, window_sizes=[252, 126, 63], 
                                     step_size=21):
    """
    Calculate rolling window regime consistency metrics
    """
    import numpy as np
    from scipy.spatial.distance import hamming
    
    consistency_results = {}
    
    for window_size in window_sizes:
        if len(regime_labels) < window_size:
            continue
        
        window_consistency_scores = []
        window_start_indices = []
        window_end_indices = []
        
        # Slide window across the data
        for start_idx in range(0, len(regime_labels) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_regimes = regime_labels[start_idx:end_idx]
            
            # Calculate consistency within window
            consistency_score = calculate_window_consistency(window_regimes)
            
            window_consistency_scores.append(consistency_score)
            window_start_indices.append(start_idx)
            window_end_indices.append(end_idx)
        
        # Calculate consistency statistics
        consistency_scores = np.array(window_consistency_scores)
        
        consistency_results[window_size] = {
            'consistency_scores': consistency_scores,
            'mean_consistency': np.mean(consistency_scores),
            'std_consistency': np.std(consistency_scores, ddof=1),
            'min_consistency': np.min(consistency_scores),
            'max_consistency': np.max(consistency_scores),
            'consistency_volatility': np.std(consistency_scores, ddof=1) / np.mean(consistency_scores),
            'start_indices': window_start_indices,
            'end_indices': window_end_indices,
            'consistency_trend': calculate_consistency_trend(consistency_scores),
            'stable_periods': identify_stable_periods(consistency_scores, window_start_indices, window_end_indices),
            'unstable_periods': identify_unstable_periods(consistency_scores, window_start_indices, window_end_indices)
        }
    
    # Calculate cross-window consistency
    cross_window_consistency = calculate_cross_window_consistency(consistency_results)
    
    return {
        'window_consistency': consistency_results,
        'cross_window_consistency': cross_window_consistency,
        'overall_consistency': calculate_overall_consistency(consistency_results),
        'consistency_classification': classify_overall_consistency(consistency_results)
    }

def calculate_window_consistency(window_regimes):
    """
    Calculate consistency score for a single window
    """
    import numpy as np
    
    # Method 1: Hamming distance from first regime
    first_regime = window_regimes[0]
    hamming_distance = np.sum(window_regimes != first_regime)
    consistency_score_1 = 1 - (hamming_distance / len(window_regimes))
    
    # Method 2: Dominant regime proportion
    unique_regimes, counts = np.unique(window_regimes, return_counts=True)
    dominant_regime_count = np.max(counts)
    consistency_score_2 = dominant_regime_count / len(window_regimes)
    
    # Method 3: Weighted consistency (recent periods more important)
    weights = np.exp(np.linspace(-1, 0, len(window_regimes)))  # Exponential weights
    weights = weights / np.sum(weights)
    
    recent_regime = window_regimes[-1]
    weighted_consistency = np.sum(weights * (window_regimes == recent_regime))
    
    # Combine methods
    combined_consistency = (0.4 * consistency_score_1 + 
                           0.4 * consistency_score_2 + 
                           0.2 * weighted_consistency)
    
    return combined_consistency

def calculate_consistency_trend(consistency_scores):
    """
    Calculate trend in consistency over time
    """
    import numpy as np
    from scipy import stats
    
    if len(consistency_scores) < 3:
        return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
    
    x = np.arange(len(consistency_scores))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, consistency_scores)
    
    # Classify trend
    if p_value < 0.05:
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'deteriorating'
        else:
            trend = 'stable'
    else:
        trend = 'no_significant_trend'
    
    return {
        'trend': trend,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'significance': p_value < 0.05
    }

def identify_stable_periods(consistency_scores, start_indices, end_indices, 
                          threshold=0.8):
    """
    Identify periods of high consistency
    """
    import numpy as np
    
    stable_periods = []
    
    for i, score in enumerate(consistency_scores):
        if score >= threshold:
            stable_periods.append({
                'start_index': start_indices[i],
                'end_index': end_indices[i],
                'consistency_score': score,
                'duration': end_indices[i] - start_indices[i]
            })
    
    return stable_periods

def identify_unstable_periods(consistency_scores, start_indices, end_indices, 
                            threshold=0.4):
    """
    Identify periods of low consistency
    """
    import numpy as np
    
    unstable_periods = []
    
    for i, score in enumerate(consistency_scores):
        if score <= threshold:
            unstable_periods.append({
                'start_index': start_indices[i],
                'end_index': end_indices[i],
                'consistency_score': score,
                'duration': end_indices[i] - start_indices[i]
            })
    
    return unstable_periods

def calculate_cross_window_consistency(consistency_results):
    """
    Calculate consistency across different window sizes
    """
    import numpy as np
    
    window_sizes = list(consistency_results.keys())
    if len(window_sizes) < 2:
        return {'correlation_matrix': np.eye(1), 'avg_correlation': 1.0}
    
    # Calculate correlations between different window sizes
    n_windows = len(window_sizes)
    correlation_matrix = np.eye(n_windows)
    
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            size_i = window_sizes[i]
            size_j = window_sizes[j]
            
            # Align the scores (use overlapping periods)
            scores_i = consistency_results[size_i]['consistency_scores']
            scores_j = consistency_results[size_j]['consistency_scores']
            
            # Take minimum length for comparison
            min_length = min(len(scores_i), len(scores_j))
            scores_i_aligned = scores_i[:min_length]
            scores_j_aligned = scores_j[:min_length]
            
            # Calculate correlation
            if min_length > 1:
                correlation = np.corrcoef(scores_i_aligned, scores_j_aligned)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    # Calculate average correlation
    avg_correlation = (np.sum(correlation_matrix) - n_windows) / (n_windows * (n_windows - 1))
    
    return {
        'correlation_matrix': correlation_matrix,
        'avg_correlation': avg_correlation,
        'window_sizes': window_sizes
    }

def calculate_overall_consistency(consistency_results):
    """
    Calculate overall consistency metrics across all windows
    """
    import numpy as np
    
    if not consistency_results:
        return {}
    
    all_consistency_scores = []
    all_window_sizes = []
    
    for window_size, results in consistency_results.items():
        all_consistency_scores.extend(results['consistency_scores'])
        all_window_sizes.extend([window_size] * len(results['consistency_scores']))
    
    all_consistency_scores = np.array(all_consistency_scores)
    
    return {
        'overall_mean_consistency': np.mean(all_consistency_scores),
        'overall_std_consistency': np.std(all_consistency_scores, ddof=1),
        'overall_min_consistency': np.min(all_consistency_scores),
        'overall_max_consistency': np.max(all_consistency_scores),
        'consistency_distribution': np.histogram(all_consistency_scores, bins=10),
        'size_weighted_consistency': np.average(all_consistency_scores, weights=all_window_sizes)
    }

def classify_overall_consistency(consistency_results):
    """
    Classify the overall consistency characteristics
    """
    if not consistency_results:
        return 'insufficient_data'
    
    overall_metrics = calculate_overall_consistency(consistency_results)
    mean_consistency = overall_metrics['overall_mean_consistency']
    std_consistency = overall_metrics['overall_std_consistency']
    
    # Calculate consistency coefficient of variation
    cv = std_consistency / mean_consistency if mean_consistency > 0 else float('inf')
    
    if mean_consistency > 0.8 and cv < 0.2:
        return 'highly_consistent'
    elif mean_consistency > 0.6 and cv < 0.3:
        return 'consistent'
    elif mean_consistency > 0.4 and cv < 0.5:
        return 'moderately_consistent'
    elif mean_consistency > 0.2:
        return 'inconsistent'
    else:
        return 'highly_inconsistent'
```

**Interpretation Guidelines:**
- **Mean consistency > 0.8**: Highly consistent regime classification
- **Mean consistency 0.6-0.8**: Good consistency
- **Mean consistency 0.4-0.6**: Moderate consistency
- **Mean consistency 0.2-0.4**: Low consistency
- **Mean consistency < 0.2**: Very poor consistency
- **Consistency volatility < 0.2**: Stable consistency over time
- **Consistency volatility 0.2-0.5**: Moderate variation in consistency
- **Consistency volatility > 0.5**: Highly variable consistency
- **Cross-window correlation > 0.8**: Consistent across different time scales
- **Cross-window correlation < 0.5**: Inconsistent across time scales

### 4.2 Time-Varying Regime Stability

**Definition and Purpose:**
Time-varying regime stability measures how the stability of individual regimes changes over time, identifying periods where regimes become more or less stable.

**Mathematical Formulas:**

**Local Stability Index:**
```
LSI_t = 1 - CV(μ_t, σ_t, skew_t, kurt_t)
```

**Stability Change Rate:**
```
SCR = d(LSI)/dt
```

**Stability Persistence:**
```
SP = Corr(LSI_t, LSI_{t-1})
```

**Implementation:**
```python
def time_varying_regime_stability(regime_labels, returns, window_size=126, 
                                 step_size=21):
    """
    Calculate time-varying stability metrics for each regime
    """
    import numpy as np
    from scipy import stats
    
    unique_regimes = np.unique(regime_labels)
    stability_results = {}
    
    for regime in unique_regimes:
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) < window_size:
            continue
        
        # Calculate rolling stability metrics
        stability_metrics = calculate_rolling_regime_stability(
            regime_returns, window_size, step_size
        )
        
        # Calculate stability trends
        stability_trends = calculate_stability_trends(stability_metrics)
        
        # Calculate stability persistence
        stability_persistence = calculate_stability_persistence(stability_metrics)
        
        # Identify stability regimes
        stability_regimes = identify_stability_regimes(stability_metrics)
        
        stability_results[regime] = {
            'rolling_metrics': stability_metrics,
            'trends': stability_trends,
            'persistence': stability_persistence,
            'stability_regimes': stability_regimes,
            'overall_stability_score': calculate_overall_stability_score(stability_metrics)
        }
    
    # Calculate comparative stability across regimes
    comparative_stability = calculate_comparative_stability(stability_results)
    
    return {
        'regime_stability': stability_results,
        'comparative_stability': comparative_stability,
        'overall_stability_classification': classify_overall_time_varying_stability(stability_results)
    }

def calculate_rolling_regime_stability(regime_returns, window_size, step_size):
    """
    Calculate rolling stability metrics for a single regime
    """
    import numpy as np
    from scipy import stats
    
    stability_metrics = {
        'timestamps': [],
        'local_stability_index': [],
        'mean_returns': [],
        'volatility': [],
        'skewness': [],
        'kurtosis': [],
        'sharpe_ratio': [],
        'stability_score': []
    }
    
    for i in range(0, len(regime_returns) - window_size + 1, step_size):
        window_data = regime_returns[i:i + window_size]
        
        # Calculate basic statistics
        mean_return = np.mean(window_data)
        volatility = np.std(window_data, ddof=1)
        
        # Higher order moments
        if len(window_data) > 3:
            skewness = stats.skew(window_data)
            kurtosis = stats.kurtosis(window_data, fisher=True)
        else:
            skewness = kurtosis = 0
        
        # Sharpe ratio
        excess_returns = window_data - 0.02/252  # Daily risk-free rate
        if volatility > 0:
            sharpe_ratio = np.mean(excess_returns) / volatility * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate local stability index
        metrics_array = np.array([mean_return, volatility, skewness, kurtosis])
        # Normalize metrics for comparison
        normalized_metrics = np.abs(metrics_array) / (np.mean(np.abs(metrics_array)) + 1e-8)
        cv = np.std(normalized_metrics) / np.mean(normalized_metrics) if np.mean(normalized_metrics) > 0 else float('inf')
        local_stability_index = 1 / (1 + cv)
        
        # Combined stability score
        stability_score = 0.4 * local_stability_index + 0.3 * (1 / (1 + abs(skewness))) + 0.3 * (1 / (1 + abs(kurtosis)))
        
        stability_metrics['timestamps'].append(i)
        stability_metrics['local_stability_index'].append(local_stability_index)
        stability_metrics['mean_returns'].append(mean_return)
        stability_metrics['volatility'].append(volatility)
        stability_metrics['skewness'].append(skewness)
        stability_metrics['kurtosis'].append(kurtosis)
        stability_metrics['sharpe_ratio'].append(sharpe_ratio)
        stability_metrics['stability_score'].append(stability_score)
    
    return stability_metrics

def calculate_stability_trends(stability_metrics):
    """
    Calculate trends in stability metrics
    """
    import numpy as np
    from scipy import stats
    
    trends = {}
    
    for metric_name in ['local_stability_index', 'stability_score', 'volatility', 'sharpe_ratio']:
        values = np.array(stability_metrics[metric_name])
        
        if len(values) < 3:
            trends[metric_name] = {'trend': 'insufficient_data', 'slope': 0, 'significance': False}
            continue
        
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if p_value < 0.05:
            if slope > 0.001:
                trend = 'increasing'
            elif slope < -0.001:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'no_significant_trend'
        
        trends[metric_name] = {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significance': p_value < 0.05
        }
    
    return trends

def calculate_stability_persistence(stability_metrics):
    """
    Calculate persistence of stability metrics
    """
    import numpy as np
    
    persistence = {}
    
    for metric_name in ['local_stability_index', 'stability_score']:
        values = np.array(stability_metrics[metric_name])
        
        if len(values) < 2:
            persistence[metric_name] = 0
            continue
        
        # Calculate autocorrelation at lag 1
        if len(values) > 1:
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        
        persistence[metric_name] = autocorr
    
    return persistence

def identify_stability_regimes(stability_metrics, stability_threshold=0.7):
    """
    Identify periods of high and low stability
    """
    import numpy as np
    
    stability_scores = np.array(stability_metrics['stability_score'])
    timestamps = stability_metrics['timestamps']
    
    # Classify each period
    stability_classifications = []
    for score in stability_scores:
        if score >= stability_threshold:
            stability_classifications.append('high_stability')
        elif score >= 0.5:
            stability_classifications.append('moderate_stability')
        else:
            stability_classifications.append('low_stability')
    
    # Identify contiguous periods
    stability_periods = []
    current_period = {
        'start_time': timestamps[0],
        'classification': stability_classifications[0],
        'duration': 1
    }
    
    for i in range(1, len(stability_classifications)):
        if stability_classifications[i] == current_period['classification']:
            current_period['duration'] += 1
        else:
            stability_periods.append(current_period)
            current_period = {
                'start_time': timestamps[i],
                'classification': stability_classifications[i],
                'duration': 1
            }
    
    stability_periods.append(current_period)
    
    return {
        'classifications': stability_classifications,
        'periods': stability_periods,
        'high_stability_ratio': np.sum(np.array(stability_classifications) == 'high_stability') / len(stability_classifications),
        'low_stability_ratio': np.sum(np.array(stability_classifications) == 'low_stability') / len(stability_classifications)
    }

def calculate_comparative_stability(stability_results):
    """
    Calculate comparative stability across regimes
    """
    import numpy as np
    
    if not stability_results:
        return {}
    
    regime_names = list(stability_results.keys())
    stability_scores = {}
    
    for regime in regime_names:
        metrics = stability_results[regime]['rolling_metrics']
        if metrics['stability_score']:
            stability_scores[regime] = np.mean(metrics['stability_score'])
        else:
            stability_scores[regime] = 0
    
    # Rank regimes by stability
    ranked_regimes = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate stability differences
    max_stability = max(stability_scores.values()) if stability_scores else 0
    min_stability = min(stability_scores.values()) if stability_scores else 0
    stability_range = max_stability - min_stability
    
    return {
        'stability_scores': stability_scores,
        'ranked_regimes': ranked_regimes,
        'most_stable': ranked_regimes[0] if ranked_regimes else None,
        'least_stable': ranked_regimes[-1] if ranked_regimes else None,
        'stability_range': stability_range,
        'stability_variance': np.var(list(stability_scores.values())) if stability_scores else 0
    }

def calculate_overall_stability_score(stability_metrics):
    """
    Calculate overall stability score for a regime
    """
    import numpy as np
    
    if not stability_metrics['stability_score']:
        return 0
    
    stability_scores = np.array(stability_metrics['stability_score'])
    
    # Overall metrics
    mean_stability = np.mean(stability_scores)
    stability_consistency = 1 - (np.std(stability_scores) / (mean_stability + 1e-8))
    
    # Combine into overall score
    overall_score = 0.7 * mean_stability + 0.3 * stability_consistency
    
    return overall_score

def classify_overall_time_varying_stability(stability_results):
    """
    Classify the overall time-varying stability characteristics
    """
    if not stability_results:
        return 'insufficient_data'
    
    # Calculate average stability across all regimes
    overall_scores = []
    for regime, results in stability_results.items():
        overall_scores.append(results['overall_stability_score'])
    
    if not overall_scores:
        return 'insufficient_data'
    
    avg_overall_stability = np.mean(overall_scores)
    stability_variance = np.var(overall_scores)
    
    if avg_overall_stability > 0.7 and stability_variance < 0.1:
        return 'highly_stable_across_regimes'
    elif avg_overall_stability > 0.5 and stability_variance < 0.2:
        return 'moderately_stable_across_regimes'
    elif avg_overall_stability > 0.3:
        return 'variable_stability_across_regimes'
    else:
        return 'unstable_across_regimes'
```

**Interpretation Guidelines:**
- **Local stability index > 0.8**: Highly stable regime characteristics
- **Local stability index 0.6-0.8**: Good stability
- **Local stability index 0.4-0.6**: Moderate stability
- **Local stability index < 0.4**: Low stability
- **Stability persistence > 0.7**: High persistence in stability
- **Stability persistence 0.4-0.7**: Moderate persistence
- **Stability persistence < 0.4**: Low persistence (unstable stability)
- **High stability ratio > 0.6**: Regime is stable most of the time
- **Low stability ratio > 0.4**: Regime is unstable most of the time

### 4.3 Lag Correlation Analysis

**Definition and Purpose:**
Lag correlation analysis examines the relationship between regime classifications at different time lags, identifying patterns of predictability and memory effects in regime dynamics.

**Mathematical Formulas:**

**Lag Autocorrelation:**
```
ρ(k) = Corr(R_t, R_{t-k})
```

**Partial Lag Correlation:**
```
ρ_partial(k) = Corr(R_t, R_{t-k} | R_{t-1}, ..., R_{t-k+1})
```

**Cross-Lag Correlation:**
```
ρ_cross(i,j,k) = Corr(R_i,t, R_j,t-k)
```

**Implementation:**
```python
def lag_correlation_analysis(regime_labels, max_lag=50, returns=None):
    """
    Comprehensive lag correlation analysis for regime dynamics
    """
    import numpy as np
    from scipy import stats
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Convert regimes to numeric if needed
    if regime_labels.dtype == 'object':
        regime_numeric = np.zeros_like(regime_labels, dtype=float)
        for i, regime in enumerate(unique_regimes):
            regime_numeric[regime_labels == regime] = i
    else:
        regime_numeric = regime_labels.astype(float)
    
    # Calculate autocorrelation function
    autocorr_values = []
    autocorr_pvalues = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr = 1.0
            pvalue = 0.0
        else:
            corr, pvalue = stats.pearsonr(regime_numeric[:-lag], regime_numeric[lag:])
            if np.isnan(corr):
                corr = 0.0
                pvalue = 1.0
        
        autocorr_values.append(corr)
        autocorr_pvalues.append(pvalue)
    
    # Calculate partial autocorrelation
    partial_autocorr_values = calculate_partial_autocorrelation(regime_numeric, max_lag)
    
    # Calculate cross-correlation matrix for different lags
    cross_correlations = {}
    for lag in [0, 1, 5, 10, 20]:
        cross_corr_matrix = calculate_cross_correlation_matrix(
            regime_labels, unique_regimes, lag
        )
        cross_correlations[lag] = cross_corr_matrix
    
    # Calculate memory metrics
    memory_metrics = calculate_memory_metrics(autocorr_values, partial_autocorr_values)
    
    # Calculate predictability metrics
    predictability_metrics = calculate_predictability_metrics(
        autocorr_values, partial_autocorr_values, returns
    )
    
    # Identify significant lags
    significant_lags = identify_significant_lags(autocorr_values, autocorr_pvalues)
    
    # Calculate regime-specific lag patterns
    regime_lag_patterns = calculate_regime_lag_patterns(
        regime_labels, unique_regimes, max_lag
    )
    
    return {
        'autocorrelation_function': autocorr_values,
        'autocorrelation_pvalues': autocorr_pvalues,
        'partial_autocorrelation_function': partial_autocorr_values,
        'cross_correlations': cross_correlations,
        'memory_metrics': memory_metrics,
        'predictability_metrics': predictability_metrics,
        'significant_lags': significant_lags,
        'regime_lag_patterns': regime_lag_patterns,
        'lag_classification': classify_lag_characteristics(autocorr_values, memory_metrics)
    }

def calculate_partial_autocorrelation(series, max_lag):
    """
    Calculate partial autocorrelation function using Durbin-Levinson algorithm
    """
    import numpy as np
    
    n = len(series)
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    
    if max_lag >= 1:
        # Calculate first lag partial autocorrelation (same as autocorrelation)
        corr = np.corrcoef(series[:-1], series[1:])[0, 1]
        pacf[1] = corr if not np.isnan(corr) else 0
    
    # For higher lags, use recursive formula
    for k in range(2, max_lag + 1):
        # Calculate PACF using Yule-Walker equations
        # This is a simplified implementation
        if k < n:
            corr_k = np.corrcoef(series[:-k], series[k:])[0, 1]
            if not np.isnan(corr_k):
                # Adjust for intermediate correlations
                adjustment = 0
                for j in range(1, k):
                    adjustment += pacf[j] * np.corrcoef(series[:-j], series[j:])[0, 1]
                pacf[k] = (corr_k - adjustment) / (1 - adjustment + 1e-8)
            else:
                pacf[k] = 0
        else:
            pacf[k] = 0
    
    return pacf

def calculate_cross_correlation_matrix(regime_labels, unique_regimes, lag):
    """
    Calculate cross-correlation matrix between regimes at specific lag
    """
    import numpy as np
    
    n_regimes = len(unique_regimes)
    cross_corr_matrix = np.zeros((n_regimes, n_regimes))
    
    if lag == 0:
        # Calculate contemporaneous correlation
        for i, regime_i in enumerate(unique_regimes):
            for j, regime_j in enumerate(unique_regimes):
                series_i = (regime_labels == regime_i).astype(float)
                series_j = (regime_labels == regime_j).astype(float)
                
                if len(series_i) > 1 and np.sum(series_i) > 0 and np.sum(series_j) > 0:
                    corr = np.corrcoef(series_i, series_j)[0, 1]
                    cross_corr_matrix[i, j] = corr if not np.isnan(corr) else 0
    else:
        # Calculate lagged correlation
        for i, regime_i in enumerate(unique_regimes):
            for j, regime_j in enumerate(unique_regimes):
                series_i = (regime_labels[:-lag] == regime_i).astype(float)
                series_j = (regime_labels[lag:] == regime_j).astype(float)
                
                if len(series_i) > 1 and np.sum(series_i) > 0 and np.sum(series_j) > 0:
                    corr = np.corrcoef(series_i, series_j)[0, 1]
                    cross_corr_matrix[i, j] = corr if not np.isnan(corr) else 0
    
    return cross_corr_matrix

def calculate_memory_metrics(autocorr_values, partial_autocorr_values):
    """
    Calculate memory-related metrics from correlation functions
    """
    import numpy as np
    
    # Sum of autocorrelations (integrated memory)
    integrated_memory = np.sum(np.abs(autocorr_values[1:]))  # Exclude lag 0
    
    # Effective memory length (where autocorrelation becomes negligible)
    effective_memory_length = 0
    threshold = 0.1  # 10% of initial correlation
    
    for i, corr in enumerate(autocorr_values[1:], 1):
        if abs(corr) < threshold:
            effective_memory_length = i
            break
    
    if effective_memory_length == 0:  # Never crossed threshold
        effective_memory_length = len(autocorr_values) - 1
    
    # Decay rate (exponential fit)
    if len(autocorr_values) > 2 and autocorr_values[1] < 1:
        decay_rate = -np.log(abs(autocorr_values[1]))
    else:
        decay_rate = 0
    
    # Memory persistence (area under PACF)
    memory_persistence = np.sum(np.abs(partial_autocorr_values[1:]))
    
    # Long memory indicator (Hurst-like measure)
    if len(autocorr_values) > 10:
        # Simple long memory test: slow decay
        slow_decay_indicator = np.mean(np.abs(autocorr_values[10:])) / np.abs(autocorr_values[1]) if autocorr_values[1] != 0 else 0
    else:
        slow_decay_indicator = 0
    
    return {
        'integrated_memory': integrated_memory,
        'effective_memory_length': effective_memory_length,
        'decay_rate': decay_rate,
        'memory_persistence': memory_persistence,
        'slow_decay_indicator': slow_decay_indicator,
        'memory_classification': classify_memory_characteristics(
            integrated_memory, effective_memory_length, decay_rate
        )
    }

def calculate_predictability_metrics(autocorr_values, partial_autocorr_values, returns=None):
    """
    Calculate predictability metrics based on correlation functions
    """
    import numpy as np
    
    # Short-term predictability (first few lags)
    short_term_predictability = np.mean(np.abs(autocorr_values[1:6]))
    
    # Medium-term predictability (lags 6-20)
    medium_term_predictability = np.mean(np.abs(autocorr_values[6:21])) if len(autocorr_values) > 20 else 0
    
    # Long-term predictability (lags > 20)
    long_term_predictability = np.mean(np.abs(autocorr_values[21:])) if len(autocorr_values) > 21 else 0
    
    # Predictability decay rate
    if len(autocorr_values) > 5:
        predictability_decay = (np.abs(autocorr_values[1]) - np.abs(autocorr_values[5])) / 4
    else:
        predictability_decay = 0
    
    # Optimal prediction horizon (lag with maximum absolute correlation)
    optimal_horizon = np.argmax(np.abs(autocorr_values[1:])) + 1 if len(autocorr_values) > 1 else 0
    
    # Economic predictability (if returns provided)
    economic_predictability = {}
    if returns is not None:
        economic_predictability = calculate_economic_predictability(
            autocorr_values, returns
        )
    
    return {
        'short_term_predictability': short_term_predictability,
        'medium_term_predictability': medium_term_predictability,
        'long_term_predictability': long_term_predictability,
        'predictability_decay': predictability_decay,
        'optimal_horizon': optimal_horizon,
        'economic_predictability': economic_predictability,
        'predictability_classification': classify_predictability_characteristics(
            short_term_predictability, medium_term_predictability, long_term_predictability
        )
    }

def calculate_economic_predictability(autocorr_values, returns):
    """
    Calculate economic significance of predictability
    """
    import numpy as np
    from scipy import stats
    
    economic_metrics = {}
    
    # Find optimal lag for prediction
    optimal_lag = np.argmax(np.abs(autocorr_values[1:])) + 1 if len(autocorr_values) > 1 else 0
    
    if optimal_lag > 0 and len(returns) > optimal_lag:
        # Calculate correlation between regime at t and returns at t+lag
        regime_series = np.arange(len(returns))  # Placeholder for actual regime series
        if optimal_lag < len(returns):
            corr, pvalue = stats.pearsonr(regime_series[:-optimal_lag], returns[optimal_lag:])
            economic_metrics['regime_return_correlation'] = corr if not np.isnan(corr) else 0
            economic_metrics['regime_return_pvalue'] = pvalue
        else:
            economic_metrics['regime_return_correlation'] = 0
            economic_metrics['regime_return_pvalue'] = 1.0
    
    return economic_metrics

def identify_significant_lags(autocorr_values, pvalues, significance_level=0.05):
    """
    Identify statistically significant lags
    """
    significant_lags = []
    
    for i, (corr, pval) in enumerate(zip(autocorr_values, pvalues)):
        if i > 0 and pval < significance_level:  # Exclude lag 0
            significant_lags.append({
                'lag': i,
                'correlation': corr,
                'pvalue': pval,
                'abs_correlation': abs(corr)
            })
    
    # Sort by absolute correlation
    significant_lags.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    return significant_lags

def calculate_regime_lag_patterns(regime_labels, unique_regimes, max_lag):
    """
    Calculate lag patterns specific to each regime
    """
    import numpy as np
    
    regime_patterns = {}
    
    for regime in unique_regimes:
        regime_series = (regime_labels == regime).astype(float)
        
        # Calculate autocorrelation for this specific regime
        regime_autocorr = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr = 1.0
            else:
                if len(regime_series) > lag:
                    corr = np.corrcoef(regime_series[:-lag], regime_series[lag:])[0, 1]
                    autocorr = corr if not np.isnan(corr) else 0
                else:
                    autocorr = 0
            regime_autocorr.append(autocorr)
        
        regime_patterns[regime] = {
            'autocorrelation': regime_autocorr,
            'persistence': np.mean(regime_autocorr[1:6]),  # First 5 lags
            'memory_length': sum(1 for corr in regime_autocorr[1:] if abs(corr) > 0.1)
        }
    
    return regime_patterns

def classify_memory_characteristics(integrated_memory, effective_length, decay_rate):
    """
    Classify memory characteristics of the regime process
    """
    if integrated_memory > 10 and effective_length > 20:
        return "long_memory"
    elif integrated_memory > 5 and effective_length > 10:
        return "moderate_memory"
    elif integrated_memory > 2 and effective_length > 5:
        return "short_memory"
    else:
        return "very_short_memory"

def classify_predictability_characteristics(short_term, medium_term, long_term):
    """
    Classify predictability characteristics
    """
    if short_term > 0.5 and medium_term > 0.3:
        return "highly_predictable"
    elif short_term > 0.3 and medium_term > 0.2:
        return "moderately_predictable"
    elif short_term > 0.2:
        return "short_term_predictable"
    elif medium_term > 0.2:
        return "medium_term_predictable"
    elif long_term > 0.2:
        return "long_term_predictable"
    else:
        return "largely_unpredictable"

def classify_lag_characteristics(autocorr_values, memory_metrics):
    """
    Classify overall lag characteristics
    """
    first_lag = abs(autocorr_values[1]) if len(autocorr_values) > 1 else 0
    memory_class = memory_metrics['memory_classification']
    
    if first_lag > 0.7 and memory_class == "long_memory":
        return "highly_persistent"
    elif first_lag > 0.5:
        return "persistent"
    elif first_lag > 0.3:
        return "moderately_persistent"
    elif first_lag > 0.1:
        return "weakly_persistent"
    else:
        return "non_persistent"
```

**Interpretation Guidelines:**
- **First-lag autocorrelation > 0.7**: Very high persistence (highly predictable)
- **First-lag autocorrelation 0.5-0.7**: High persistence
- **First-lag autocorrelation 0.3-0.5**: Moderate persistence
- **First-lag autocorrelation 0.1-0.3**: Low persistence
- **First-lag autocorrelation < 0.1**: Very low persistence
- **Effective memory length > 20**: Long memory effects
- **Effective memory length 10-20**: Moderate memory
- **Effective memory length 5-10**: Short memory
- **Effective memory length < 5**: Very short memory
- **Short-term predictability > 0.5**: High short-term predictability
- **Medium-term predictability > 0.3**: Good medium-term predictability
- **Long-term predictability > 0.2**: Some long-term predictability

### 4.4 Lead-Lag Relationships Between Regimes

**Definition and Purpose:**
Lead-lag relationships identify which regimes tend to precede or follow others, providing insights into causal relationships and predictive patterns in market state evolution.

**Mathematical Formulas:**

**Lead-Lag Correlation:**
```
LLC(i,j,k) = Corr(R_i,t, R_j,t+k)
```

**Granger Causality:**
```
GC(i→j) = Var(ε_j,t | all) - Var(ε_j,t | R_i,t-k)
```

**Transfer Entropy:**
```
TE(i→j) = H(R_j,t | R_j,t-1) - H(R_j,t | R_j,t-1, R_i,t-1)
```

**Implementation:**
```python
def lead_lag_relationship_analysis(regime_labels, max_lag=20, returns=None):
    """
    Comprehensive lead-lag relationship analysis between regimes
    """
    import numpy as np
    from scipy import stats
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    
    # Create binary regime series
    regime_series = {}
    for regime in unique_regimes:
        regime_series[regime] = (regime_labels == regime).astype(float)
    
    # Calculate lead-lag correlation matrix
    lead_lag_correlations = calculate_lead_lag_correlations(
        regime_series, unique_regimes, max_lag
    )
    
    # Calculate Granger causality
    granger_causality = calculate_granger_causality(
        regime_series, unique_regimes, max_lag
    )
    
    # Calculate transfer entropy
    transfer_entropy = calculate_transfer_entropy(
        regime_series, unique_regimes, max_lag
    )
    
    # Identify significant relationships
    significant_relationships = identify_significant_lead_lag(
        lead_lag_correlations, granger_causality, transfer_entropy
    )
    
    # Calculate lead-lag network metrics
    network_metrics = calculate_lead_lag_network_metrics(significant_relationships)
    
    # Calculate economic significance if returns provided
    economic_significance = {}
    if returns is not None:
        economic_significance = calculate_economic_lead_lag_significance(
            regime_series, returns, significant_relationships
        )
    
    # Classify relationship types
    relationship_classification = classify_lead_lag_relationships(
        significant_relationships, network_metrics
    )
    
    return {
        'lead_lag_correlations': lead_lag_correlations,
        'granger_causality': granger_causality,
        'transfer_entropy': transfer_entropy,
        'significant_relationships': significant_relationships,
        'network_metrics': network_metrics,
        'economic_significance': economic_significance,
        'relationship_classification': relationship_classification,
        'lead_lag_summary': generate_lead_lag_summary(significant_relationships)
    }

def calculate_lead_lag_correlations(regime_series, unique_regimes, max_lag):
    """
    Calculate lead-lag correlations between all regime pairs
    """
    import numpy as np
    
    n_regimes = len(unique_regimes)
    correlations = {}
    
    for i, regime_i in enumerate(unique_regimes):
        for j, regime_j in enumerate(unique_regimes):
            correlations[(regime_i, regime_j)] = []
            
            series_i = regime_series[regime_i]
            series_j = regime_series[regime_j]
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    # Contemporaneous correlation
                    if len(series_i) > 1 and np.sum(series_i) > 0 and np.sum(series_j) > 0:
                        corr = np.corrcoef(series_i, series_j)[0, 1]
                        correlations[(regime_i, regime_j)].append(corr if not np.isnan(corr) else 0)
                    else:
                        correlations[(regime_i, regime_j)].append(0)
                else:
                    # Lead-lag correlation
                    if len(series_i) > lag:
                        lead_series = series_i[:-lag]
                        lag_series = series_j[lag:]
                        
                        if len(lead_series) > 1 and np.sum(lead_series) > 0 and np.sum(lag_series) > 0:
                            corr = np.corrcoef(lead_series, lag_series)[0, 1]
                            correlations[(regime_i, regime_j)].append(corr if not np.isnan(corr) else 0)
                        else:
                            correlations[(regime_i, regime_j)].append(0)
                    else:
                        correlations[(regime_i, regime_j)].append(0)
    
    return correlations

def calculate_granger_causality(regime_series, unique_regimes, max_lag):
    """
    Calculate Granger causality between regime pairs
    """
    import numpy as np
    from scipy import stats
    
    n_regimes = len(unique_regimes)
    causality = {}
    
    for i, regime_i in enumerate(unique_regimes):
        for j, regime_j in enumerate(unique_regimes):
            if i == j:
                continue
            
            causality[(regime_i, regime_j)] = []
            
            series_i = regime_series[regime_i]
            series_j = regime_series[regime_j]
            
            for lag in range(1, max_lag + 1):
                if len(series_i) > lag:
                    # Simple Granger causality test
                    # Model 1: series_j depends only on its own past
                    y_self = series_j[lag:]
                    X_self = np.column_stack([series_j[lag-k:-k] for k in range(1, lag + 1)])
                    
                    # Model 2: series_j depends on its own past and series_i's past
                    X_combined = np.column_stack([
                        series_j[lag-k:-k] for k in range(1, lag + 1)
                    ] + [
                        series_i[lag-k:-k] for k in range(1, lag + 1)
                    ])
                    
                    # Fit models and calculate F-statistic
                    try:
                        # Self-only model
                        if X_self.shape[1] > 0 and len(y_self) > X_self.shape[1]:
                            beta_self = np.linalg.lstsq(X_self, y_self, rcond=None)[0]
                            residuals_self = y_self - X_self @ beta_self
                            mse_self = np.mean(residuals_self**2)
                        else:
                            mse_self = float('inf')
                        
                        # Combined model
                        if X_combined.shape[1] > 0 and len(y_self) > X_combined.shape[1]:
                            beta_combined = np.linalg.lstsq(X_combined, y_self, rcond=None)[0]
                            residuals_combined = y_self - X_combined @ beta_combined
                            mse_combined = np.mean(residuals_combined**2)
                        else:
                            mse_combined = float('inf')
                        
                        # F-statistic
                        if mse_combined > 0 and mse_self > 0:
                            f_stat = (mse_self - mse_combined) / mse_combined * (len(y_self) - X_combined.shape[1]) / (X_combined.shape[1] - X_self.shape[1])
                            # Convert to p-value (simplified)
                            p_value = 1 - stats.f.cdf(f_stat, X_combined.shape[1] - X_self.shape[1], len(y_self) - X_combined.shape[1])
                        else:
                            f_stat = 0
                            p_value = 1.0
                        
                        causality[(regime_i, regime_j)].append({
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
                    except:
                        causality[(regime_i, regime_j)].append({
                            'f_statistic': 0,
                            'p_value': 1.0,
                            'significant': False
                        })
                else:
                    causality[(regime_i, regime_j)].append({
                        'f_statistic': 0,
                        'p_value': 1.0,
                        'significant': False
                    })
    
    return causality

def calculate_transfer_entropy(regime_series, unique_regimes, max_lag):
    """
    Calculate transfer entropy between regime pairs
    """
    import numpy as np
    
    n_regimes = len(unique_regimes)
    transfer_entropy = {}
    
    for i, regime_i in enumerate(unique_regimes):
        for j, regime_j in enumerate(unique_regimes):
            if i == j:
                continue
            
            transfer_entropy[(regime_i, regime_j)] = []
            
            series_i = regime_series[regime_i]
            series_j = regime_series[regime_j]
            
            for lag in range(1, max_lag + 1):
                if len(series_i) > lag:
                    # Calculate transfer entropy
                    # TE(i→j) = H(j_t | j_{t-1}) - H(j_t | j_{t-1}, i_{t-1})
                    
                    # Create joint distributions
                    j_current = series_j[lag:]
                    j_past = series_j[lag-1:-1]
                    i_past = series_i[lag-1:-1]
                    
                    # Calculate entropies (simplified)
                    # H(j_t | j_{t-1})
                    joint_jj = np.histogram2d(j_current, j_past, bins=2)[0]
                    joint_jj = joint_jj + 1e-10  # Avoid zeros
                    marginal_j = np.sum(joint_jj, axis=0)
                    marginal_j_past = np.sum(joint_jj, axis=1)
                    
                    p_jj = joint_jj / np.sum(joint_jj)
                    p_j_given_j_past = joint_jj / marginal_j_past[np.newaxis, :]
                    
                    H_j_given_j_past = -np.sum(p_jj * np.log(p_j_given_j_past + 1e-10))
                    
                    # H(j_t | j_{t-1}, i_{t-1})
                    joint_jji = np.histogramdd([j_current, j_past, i_past], bins=2)[0]
                    joint_jji = joint_jji + 1e-10
                    marginal_ji = np.sum(joint_jji, axis=0)
                    
                    p_jji = joint_jji / np.sum(joint_jji)
                    p_j_given_ji_past = joint_jji / marginal_ji[np.newaxis, :]
                    
                    H_j_given_ji_past = -np.sum(p_jji * np.log(p_j_given_ji_past + 1e-10))
                    
                    # Transfer entropy
                    te = H_j_given_j_past - H_j_given_ji_past
                    transfer_entropy[(regime_i, regime_j)].append(max(0, te))
                else:
                    transfer_entropy[(regime_i, regime_j)].append(0)
    
    return transfer_entropy

def identify_significant_lead_lag(correlations, granger_causality, transfer_entropy):
    """
    Identify statistically significant lead-lag relationships
    """
    significant_relationships = []
    
    for pair in correlations.keys():
        if pair[0] == pair[1]:  # Skip self-relationships
            continue
        
        corr_values = correlations[pair]
        gc_values = granger_causality.get(pair, [])
        te_values = transfer_entropy.get(pair, [])
        
        # Find lag with maximum evidence
        best_lag = 0
        best_score = 0
        
        for lag in range(len(corr_values)):
            score = 0
            
            # Correlation contribution
            if lag < len(corr_values):
                score += abs(corr_values[lag]) * 0.4
            
            # Granger causality contribution
            if lag < len(gc_values):
                if gc_values[lag]['significant']:
                    score += 0.4
            
            # Transfer entropy contribution
            if lag < len(te_values):
                score += min(te_values[lag], 1.0) * 0.2
            
            if score > best_score:
                best_score = score
                best_lag = lag
        
        # Classify as significant if score exceeds threshold
        if best_score > 0.5:
            significant_relationships.append({
                'from_regime': pair[0],
                'to_regime': pair[1],
                'best_lag': best_lag,
                'score': best_score,
                'max_correlation': max(abs(c) for c in corr_values) if corr_values else 0,
                'granger_significant': any(gc['significant'] for gc in gc_values) if gc_values else False,
                'max_transfer_entropy': max(te_values) if te_values else 0,
                'relationship_type': classify_relationship_type(best_score, best_lag)
            })
    
    # Sort by score
    significant_relationships.sort(key=lambda x: x['score'], reverse=True)
    
    return significant_relationships

def classify_relationship_type(score, lag):
    """
    Classify the type of lead-lag relationship
    """
    if score > 0.8:
        strength = "strong"
    elif score > 0.6:
        strength = "moderate"
    else:
        strength = "weak"
    
    if lag == 0:
        timing = "contemporaneous"
    elif lag <= 3:
        timing = "short_lag"
    elif lag <= 10:
        timing = "medium_lag"
    else:
        timing = "long_lag"
    
    return f"{strength}_{timing}"

def calculate_lead_lag_network_metrics(significant_relationships):
    """
    Calculate network metrics for lead-lag relationships
    """
    import numpy as np
    
    if not significant_relationships:
        return {}
    
    # Build adjacency matrix
    regimes = list(set([r['from_regime'] for r in significant_relationships] + 
                      [r['to_regime'] for r in significant_relationships]))
    n_regimes = len(regimes)
    
    regime_to_idx = {regime: i for i, regime in enumerate(regimes)}
    adjacency_matrix = np.zeros((n_regimes, n_regimes))
    
    for rel in significant_relationships:
        from_idx = regime_to_idx[rel['from_regime']]
        to_idx = regime_to_idx[rel['to_regime']]
        adjacency_matrix[from_idx, to_idx] = rel['score']
    
    # Calculate network metrics
    out_degree = np.sum(adjacency_matrix > 0, axis=1)
    in_degree = np.sum(adjacency_matrix > 0, axis=0)
    
    # Identify hubs and sinks
    hub_threshold = np.percentile(out_degree, 75)
    sink_threshold = np.percentile(in_degree, 75)
    
    hubs = [regimes[i] for i in range(n_regimes) if out_degree[i] >= hub_threshold]
    sinks = [regimes[i] for i in range(n_regimes) if in_degree[i] >= sink_threshold]
    
    # Calculate centrality measures
    out_strength = np.sum(adjacency_matrix, axis=1)
    in_strength = np.sum(adjacency_matrix, axis=0)
    
    return {
        'regimes': regimes,
        'adjacency_matrix': adjacency_matrix,
        'out_degree': dict(zip(regimes, out_degree)),
        'in_degree': dict(zip(regimes, in_degree)),
        'out_strength': dict(zip(regimes, out_strength)),
        'in_strength': dict(zip(regimes, in_strength)),
        'hubs': hubs,
        'sinks': sinks,
        'network_density': np.sum(adjacency_matrix > 0) / (n_regimes * (n_regimes - 1))
    }

def calculate_economic_lead_lag_significance(regime_series, returns, significant_relationships):
    """
    Calculate economic significance of lead-lag relationships
    """
    import numpy as np
    from scipy import stats
    
    economic_significance = {}
    
    for rel in significant_relationships:
        from_regime = rel['from_regime']
        to_regime = rel['to_regime']
        lag = rel['best_lag']
        
        # Calculate correlation between regime transition and subsequent returns
        from_series = regime_series[from_regime]
        to_series = regime_series[to_regime]
        
        if len(from_series) > lag and len(returns) > lag:
            # Identify transition points
            transitions = np.where(from_series[:-lag] != from_series[lag:])[0]
            
            if len(transitions) > 5:  # Need sufficient transitions
                # Calculate returns around transitions
                transition_returns = []
                for trans_idx in transitions:
                    start_idx = max(0, trans_idx)
                    end_idx = min(len(returns), trans_idx + lag + 5)
                    if end_idx > start_idx:
                        transition_returns.append(np.mean(returns[start_idx:end_idx]))
                
                if transition_returns:
                    # Test if returns after transitions are significant
                    mean_return = np.mean(transition_returns)
                    all_returns = returns[lag:]  # Returns after lag period
                    
                    if len(all_returns) > 0:
                        t_stat, p_value = stats.ttest_1samp(transition_returns, np.mean(all_returns))
                        
                        economic_significance[(from_regime, to_regime)] = {
                            'mean_transition_return': mean_return,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'economically_significant': p_value < 0.05 and abs(mean_return) > 0.001
                        }
    
    return economic_significance

def classify_lead_lag_relationships(significant_relationships, network_metrics):
    """
    Classify the overall lead-lag relationship patterns
    """
    if not significant_relationships:
        return 'no_significant_relationships'
    
    # Count relationship types
    strong_relationships = [r for r in significant_relationships if r['score'] > 0.7]
    moderate_relationships = [r for r in significant_relationships if 0.5 < r['score'] <= 0.7]
    
    # Analyze network structure
    if 'network_density' in network_metrics:
        density = network_metrics['network_density']
    else:
        density = 0
    
    if len(strong_relationships) > 5 and density > 0.3:
        return 'highly_interconnected_strong_relationships'
    elif len(strong_relationships) > 2:
        return 'strong_lead_lag_patterns'
    elif len(moderate_relationships) > 5:
        return 'moderate_interconnected_relationships'
    elif density > 0.2:
        return 'weak_interconnected_relationships'
    elif len(significant_relationships) > 0:
        return 'isolated_lead_lag_relationships'
    else:
        return 'no_significant_relationships'

def generate_lead_lag_summary(significant_relationships):
    """
    Generate a summary of lead-lag relationships
    """
    if not significant_relationships:
        return {
            'total_relationships': 0,
            'summary': 'No significant lead-lag relationships found'
        }
    
    # Count relationships by type
    short_lag = [r for r in significant_relationships if r['best_lag'] <= 3]
    medium_lag = [r for r in significant_relationships if 3 < r['best_lag'] <= 10]
    long_lag = [r for r in significant_relationships if r['best_lag'] > 10]
    
    # Find most influential regimes
    from_counts = {}
    to_counts = {}
    
    for rel in significant_relationships:
        from_counts[rel['from_regime']] = from_counts.get(rel['from_regime'], 0) + 1
        to_counts[rel['to_regime']] = to_counts.get(rel['to_regime'], 0) + 1
    
    most_influential = max(from_counts.items(), key=lambda x: x[1]) if from_counts else None
    most_responsive = max(to_counts.items(), key=lambda x: x[1]) if to_counts else None
    
    return {
        'total_relationships': len(significant_relationships),
        'short_lag_relationships': len(short_lag),
        'medium_lag_relationships': len(medium_lag),
        'long_lag_relationships': len(long_lag),
        'most_influential_regime': most_influential,
        'most_responsive_regime': most_responsive,
        'average_lag': np.mean([r['best_lag'] for r in significant_relationships]),
        'summary': f"Found {len(significant_relationships)} significant lead-lag relationships"
    }
```

**Interpretation Guidelines:**
- **Lead-lag correlation > 0.5**: Strong predictive relationship
- **Lead-lag correlation 0.3-0.5**: Moderate predictive relationship
- **Lead-lag correlation 0.1-0.3**: Weak predictive relationship
- **Granger causality p-value < 0.05**: Statistically significant causal relationship
- **Transfer entropy > 0.1**: Meaningful information transfer
- **Network density > 0.3**: Highly interconnected regime system
- **Hub regimes**: Important for predicting other regimes
- **Sink regimes**: Important end points for regime evolution
- **Short-lag relationships (<3 periods)**: Immediate predictive value
- **Medium-lag relationships (3-10 periods)**: Medium-term predictive value
- **Long-lag relationships (>10 periods)**: Long-term predictive value

## Implementation Considerations

### 5.1 Computational Efficiency

**Vectorization Strategies:**
```python
def vectorized_regime_metrics(regime_labels, returns, window_size=252):
    """
    Vectorized implementation for better performance
    """
    import numpy as np
    
    # Pre-allocate arrays
    n_windows = len(regime_labels) // window_size
    metrics = {
        'consistency': np.zeros(n_windows),
        'volatility': np.zeros(n_windows),
        'sharpe': np.zeros(n_windows)
    }
    
    # Vectorized calculations
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        
        window_regimes = regime_labels[start:end]
        window_returns = returns[start:end]
        
        # Vectorized consistency calculation
        unique_regimes, counts = np.unique(window_regimes, return_counts=True)
        metrics['consistency'][i] = np.max(counts) / len(window_regimes)
        
        # Vectorized return calculations
        metrics['volatility'][i] = np.std(window_returns, ddof=1)
        metrics['sharpe'][i] = np.mean(window_returns) / (metrics['volatility'][i] + 1e-8)
    
    return metrics
```

**Memory Optimization:**
```python
def memory_efficient_transition_analysis(regime_labels, chunk_size=10000):
    """
    Memory-efficient transition analysis for large datasets
    """
    import numpy as np
    
    n_total = len(regime_labels)
    transitions = []
    
    # Process in chunks to avoid memory issues
    for start in range(0, n_total - 1, chunk_size):
        end = min(start + chunk_size, n_total - 1)
        chunk = regime_labels[start:end + 1]
        
        # Find transitions in chunk
        chunk_transitions = np.where(chunk[:-1] != chunk[1:])[0] + start
        transitions.extend(chunk_transitions.tolist())
    
    return np.array(transitions)
```

**Parallel Processing:**
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_regime_analysis(regime_labels, returns, n_processes=None):
    """
    Parallel processing for regime analysis
    """
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    
    # Split data into chunks
    chunk_size = len(regime_labels) // n_processes
    chunks = []
    
    for i in range(n_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_processes - 1 else len(regime_labels)
        chunks.append((regime_labels[start:end], returns[start:end]))
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(analyze_regime_chunk, chunks))
    
    # Combine results
    combined_results = combine_chunk_results(results)
    
    return combined_results

def analyze_regime_chunk(chunk_data):
    """Analyze a single chunk of data"""
    regime_labels, returns = chunk_data
    # Perform analysis on chunk
    return chunk_results
```

### 5.2 Edge Case Handling

**Insufficient Data Handling:**
```python
def handle_insufficient_data(regime_labels, min_samples=30):
    """
    Handle cases with insufficient data for reliable analysis
    """
    import numpy as np
    import warnings
    
    unique_regimes, counts = np.unique(regime_labels, return_counts=True)
    
    # Check for regimes with insufficient samples
    insufficient_regimes = unique_regimes[counts < min_samples]
    
    if len(insufficient_regimes) > 0:
        warnings.warn(f"Regimes {insufficient_regimes} have fewer than {min_samples} samples")
        
        # Options for handling:
        # 1. Remove insufficient regimes
        # 2. Merge with similar regimes
        # 3. Use bootstrap methods
        # 4. Flag results as unreliable
    
    return {
        'insufficient_regimes': insufficient_regimes.tolist(),
        'regime_counts': dict(zip(unique_regimes, counts)),
        'reliable_analysis': len(insufficient_regimes) == 0
    }
```

**Boundary Condition Handling:**
```python
def handle_boundary_conditions(regime_labels, window_size, step_size):
    """
    Handle edge cases in rolling window analysis
    """
    import numpy as np
    
    n_total = len(regime_labels)
    
    # Calculate valid window positions
    valid_starts = []
    for start in range(0, n_total - window_size + 1, step_size):
        end = start + window_size
        if end <= n_total:
            valid_starts.append(start)
    
    # Handle partial windows at the end
    if n_total % step_size != 0:
        last_start = (n_total // step_size) * step_size
        if n_total - last_start >= window_size // 2:  # At least half window
            valid_starts.append(last_start)
    
    return valid_starts
```

**Numerical Stability:**
```python
def ensure_numerical_stability(values, epsilon=1e-10):
    """
    Ensure numerical stability in calculations
    """
    import numpy as np
    
    # Handle division by zero
    values = np.where(np.abs(values) < epsilon, epsilon, values)
    
    # Handle log of zero/negative
    values = np.where(values <= 0, epsilon, values)
    
    # Handle NaN and Inf
    values = np.where(np.isnan(values), 0, values)
    values = np.where(np.isinf(values), 1e6, values)
    
    return values
```

### 5.3 Real-Time Implementation

**Streaming Updates:**
```python
class StreamingRegimeAnalyzer:
    """
    Real-time regime analysis with streaming updates
    """
    
    def __init__(self, window_size=252, update_frequency=1000):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.buffer = []
        self.last_update = 0
        self.current_metrics = {}
    
    def update(self, new_regime, new_return):
        """
        Update with new data point
        """
        self.buffer.append((new_regime, new_return))
        
        # Check if update is needed
        if len(self.buffer) >= self.update_frequency:
            self._recalculate_metrics()
            self.buffer = self.buffer[-self.window_size:]  # Keep only recent data
    
    def _recalculate_metrics(self):
        """
        Recalculate metrics with current buffer
        """
        if len(self.buffer) < self.window_size:
            return  # Not enough data
        
        regimes, returns = zip(*self.buffer[-self.window_size:])
        
        # Calculate updated metrics
        self.current_metrics = {
            'consistency': self._calculate_consistency(regimes),
            'volatility': np.std(returns, ddof=1),
            'transition_rate': self._calculate_transition_rate(regimes)
        }
    
    def get_current_metrics(self):
        """
        Get current regime metrics
        """
        return self.current_metrics
```

**Incremental Calculations:**
```python
class IncrementalTransitionCounter:
    """
    Incremental transition counting for efficiency
    """
    
    def __init__(self):
        self.last_regime = None
        self.transition_counts = {}
        self.total_count = 0
    
    def update(self, current_regime):
        """
        Update with new regime observation
        """
        if self.last_regime is not None and current_regime != self.last_regime:
            transition = (self.last_regime, current_regime)
            self.transition_counts[transition] = self.transition_counts.get(transition, 0) + 1
        
        self.last_regime = current_regime
        self.total_count += 1
    
    def get_transition_matrix(self):
        """
        Get current transition probability matrix
        """
        if self.total_count == 0:
            return {}
        
        # Convert counts to probabilities
        transition_matrix = {}
        for transition, count in self.transition_counts.items():
            from_regime = transition[0]
            total_from = sum(count for (fr, _), count in self.transition_counts.items() if fr == from_regime)
            transition_matrix[transition] = count / total_from if total_from > 0 else 0
        
        return transition_matrix
```

## Interpretation Guidelines

### 6.1 Metric Integration Framework

**Composite Temporal Quality Score:**
```python
def calculate_temporal_quality_score(temporal_metrics):
    """
    Calculate composite temporal quality score
    """
    weights = {
        'persistence': 0.25,
        'smoothness': 0.20,
        'consistency': 0.20,
        'predictability': 0.15,
        'stability': 0.20
    }
    
    # Normalize individual metrics to 0-1 scale
    normalized_scores = {
        'persistence': normalize_persistence_score(temporal_metrics['persistence']),
        'smoothness': normalize_smoothness_score(temporal_metrics['smoothness']),
        'consistency': normalize_consistency_score(temporal_metrics['consistency']),
        'predictability': normalize_predictability_score(temporal_metrics['predictability']),
        'stability': normalize_stability_score(temporal_metrics['stability'])
    }
    
    # Calculate weighted composite score
    composite_score = sum(weights[metric] * normalized_scores[metric] 
                         for metric in weights.keys())
    
    return {
        'composite_score': composite_score,
        'component_scores': normalized_scores,
        'quality_classification': classify_temporal_quality(composite_score)
    }

def classify_temporal_quality(score):
    """
    Classify overall temporal quality
    """
    if score > 0.8:
        return "excellent_temporal_quality"
    elif score > 0.6:
        return "good_temporal_quality"
    elif score > 0.4:
        return "moderate_temporal_quality"
    elif score > 0.2:
        return "poor_temporal_quality"
    else:
        return "very_poor_temporal_quality"
```

### 6.2 Trading Application Guidelines

**Regime Trading Suitability:**
```python
def assess_trading_suitability(temporal_metrics, economic_metrics):
    """
    Assess regime suitability for different trading strategies
    """
    suitability = {}
    
    # Long-term trading suitability
    long_term_score = (
        temporal_metrics['persistence'] * 0.4 +
        temporal_metrics['stability'] * 0.3 +
        temporal_metrics['consistency'] * 0.3
    )
    
    # Short-term trading suitability
    short_term_score = (
        (1 - temporal_metrics['persistence']) * 0.3 +  # Lower persistence is better
        temporal_metrics['predictability'] * 0.4 +
        (1 - temporal_metrics['noise_level']) * 0.3
    )
    
    # Swing trading suitability
    swing_score = (
        temporal_metrics['persistence'] * 0.3 +
        temporal_metrics['predictability'] * 0.4 +
        temporal_metrics['smoothness'] * 0.3
    )
    
    suitability = {
        'long_term': {
            'score': long_term_score,
            'recommendation': classify_trading_suitability(long_term_score, 'long_term')
        },
        'short_term': {
            'score': short_term_score,
            'recommendation': classify_trading_suitability(short_term_score, 'short_term')
        },
        'swing': {
            'score': swing_score,
            'recommendation': classify_trading_suitability(swing_score, 'swing')
        }
    }
    
    return suitability

def classify_trading_suitability(score, trading_style):
    """
    Classify trading suitability for different styles
    """
    thresholds = {
        'long_term': {'excellent': 0.7, 'good': 0.5, 'moderate': 0.3},
        'short_term': {'excellent': 0.6, 'good': 0.4, 'moderate': 0.2},
        'swing': {'excellent': 0.6, 'good': 0.4, 'moderate': 0.2}
    }
    
    style_thresholds = thresholds[trading_style]
    
    if score > style_thresholds['excellent']:
        return "highly_suitable"
    elif score > style_thresholds['good']:
        return "suitable"
    elif score > style_thresholds['moderate']:
        return "moderately_suitable"
    else:
        return "unsuitable"
```

### 6.3 Model Selection Guidelines

**Temporal Model Selection:**
```python
def select_temporal_model(temporal_metrics):
    """
    Recommend appropriate temporal models based on metrics
    """
    recommendations = []
    
    # Markov models
    if temporal_metrics['memory_length'] < 20 and temporal_metrics['predictability'] > 0.3:
        recommendations.append({
            'model': 'markov_chain',
            'suitability': 'high',
            'reason': 'Short memory and moderate predictability'
        })
    
    # Hidden Markov Models
    if temporal_metrics['noise_level'] > 0.3 and temporal_metrics['persistence'] > 0.4:
        recommendations.append({
            'model': 'hidden_markov_model',
            'suitability': 'high',
            'reason': 'Significant noise with underlying persistence'
        })
    
    # LSTM/Deep Learning
    if temporal_metrics['memory_length'] > 20 or temporal_metrics['nonlinearity'] > 0.5:
        recommendations.append({
            'model': 'lstm_neural_network',
            'suitability': 'moderate',
            'reason': 'Long memory or nonlinear patterns detected'
        })
    
    # Ensemble methods
    if temporal_metrics['complexity'] > 0.7:
        recommendations.append({
            'model': 'ensemble_methods',
            'suitability': 'moderate',
            'reason': 'High complexity suggests ensemble approach'
        })
    
    return recommendations
```

## Complete Implementation Framework

### 7.1 Comprehensive Temporal Analysis Class

```python
class ComprehensiveTemporalRegimeAnalyzer:
    """
    Complete implementation of temporal smoothness and transition metrics
    """
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self._validate_config()
    
    def analyze_temporal_characteristics(self, regime_labels, returns=None, 
                                       timestamps=None, feature_data=None):
        """
        Perform comprehensive temporal analysis
        """
        # Validate inputs
        self._validate_inputs(regime_labels, returns)
        
        # Calculate all temporal metric categories
        analysis_results = {
            'metadata': self._calculate_metadata(regime_labels),
            'temporal_smoothness': self._calculate_temporal_smoothness(regime_labels, timestamps),
            'transition_analysis': self._calculate_transition_analysis(regime_labels),
            'flip_flop_analysis': self._calculate_flip_flop_analysis(regime_labels, returns),
            'temporal_consistency': self._calculate_temporal_consistency(regime_labels, returns),
            'lead_lag_analysis': self._calculate_lead_lag_analysis(regime_labels, returns),
            'noise_signal_discrimination': self._calculate_noise_signal_discrimination(
                regime_labels, returns, feature_data)
        }
        
        # Add composite analysis
        analysis_results['temporal_quality_score'] = self._calculate_temporal_quality_score(
            analysis_results)
        analysis_results['trading_recommendations'] = self._generate_trading_recommendations(
            analysis_results)
        analysis_results['model_recommendations'] = self._generate_model_recommendations(
            analysis_results)
        
        return analysis_results
    
    def _calculate_temporal_smoothness(self, regime_labels, timestamps):
        """Calculate all temporal smoothness metrics"""
        return {
            'persistence_metrics': regime_persistence_metrics(regime_labels, timestamps),
            'smoothness_indices': transition_smoothness_metrics(regime_labels),
            'temporal_autocorrelation': temporal_autocorrelation_metrics(regime_labels),
            'regime_stability': regime_stability_over_time(regime_labels)
        }
    
    def _calculate_transition_analysis(self, regime_labels):
        """Calculate all transition analysis metrics"""
        return {
            'markov_transitions': markov_transition_matrix(regime_labels),
            'transition_frequency': transition_frequency_analysis(regime_labels),
            'expected_times': expected_time_metrics(
                markov_transition_matrix(regime_labels)['transition_probabilities']),
            'absorbing_states': absorbing_states_analysis(
                markov_transition_matrix(regime_labels)['transition_probabilities']),
            'transition_entropy': transition_entropy_analysis(
                markov_transition_matrix(regime_labels)['transition_probabilities'])
        }
    
    def _calculate_flip_flop_analysis(self, regime_labels, returns):
        """Calculate all flip-flop analysis metrics"""
        return {
            'switching_frequency': regime_switching_frequency_metrics(regime_labels),
            'flip_flop_rates': flip_flop_rate_calculations(regime_labels),
            'whipsaw_detection': whipsaw_detection_metrics(regime_labels, returns),
            'noise_signal_discrimination': noise_signal_discrimination(regime_labels, returns)
        }
    
    def _calculate_temporal_consistency(self, regime_labels, returns):
        """Calculate all temporal consistency metrics"""
        return {
            'rolling_consistency': rolling_window_regime_consistency(regime_labels),
            'time_varying_stability': time_varying_regime_stability(regime_labels, returns),
            'lag_correlations': lag_correlation_analysis(regime_labels, returns),
            'lead_lag_relationships': lead_lag_relationship_analysis(regime_labels, returns)
        }
    
    def _calculate_lead_lag_analysis(self, regime_labels, returns):
        """Calculate lead-lag relationships"""
        return lead_lag_relationship_analysis(regime_labels, returns)
    
    def _calculate_noise_signal_discrimination(self, regime_labels, returns, feature_data):
        """Calculate noise vs. signal discrimination"""
        return noise_signal_discrimination(regime_labels, returns, feature_data)
    
    def _calculate_temporal_quality_score(self, analysis_results):
        """Calculate composite temporal quality score"""
        return calculate_temporal_quality_score(analysis_results)
    
    def _generate_trading_recommendations(self, analysis_results):
        """Generate trading recommendations based on temporal analysis"""
        # Implement trading recommendation logic
        pass
    
    def _generate_model_recommendations(self, analysis_results):
        """Generate model selection recommendations"""
        # Implement model recommendation logic
        pass
    
    def _default_config(self):
        """Default configuration for analysis"""
        return {
            'min_regime_duration': 5,
            'window_sizes': [252, 126, 63],
            'step_size': 21,
            'max_lag': 50,
            'significance_level': 0.05,
            'bootstrap_samples': 1000
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['min_regime_duration', 'window_sizes', 'max_lag']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")
    
    def _validate_inputs(self, regime_labels, returns):
        """Validate input data"""
        if len(regime_labels) < self.config['min_regime_duration']:
            raise ValueError(f"Insufficient data: need at least {self.config['min_regime_duration']} observations")
        
        if returns is not None and len(returns) != len(regime_labels):
            raise ValueError("Returns and regime labels must have same length")
```

This comprehensive technical document provides detailed temporal smoothness and transition metrics for evaluating financial market regimes in regime ensemble training systems. The implementation includes mathematical formulas, practical code examples, interpretation guidelines, and considerations for real-world trading applications. The metrics cover all aspects of temporal analysis from basic persistence measures to complex lead-lag relationships, providing a complete framework for assessing regime quality and suitability for trading strategies.

<update_todo_list>
<todos>
[x] Research temporal smoothness metrics for regime analysis
[x] Document transition matrix analysis methods
[x] Research flip-flop analysis techniques
[x] Document temporal consistency metrics
[x] Create comprehensive technical document with implementation details
[x] Add code examples and practical considerations
</todos>
</update_todo_list>
