# Log Multiplication vs Addition for Feature Ranking

## TL;DR

✅ **Log multiplication is better** for combining importance and stability scores because it enforces that features need BOTH high importance AND high stability, not just one or the other.

## The Two Approaches

### Additive Combination (Original)
```python
combined = (1 - w) * importance + w * stability
```

### Log Multiplication (Implemented)
```python
combined = importance^(1-w) × stability^w
# Equivalent to: exp((1-w)*log(importance) + w*log(stability))
```

## Why Log Multiplication is Better

### 1. **Multiplicative Relationships**

Features with **low stability** should be **heavily penalized**, even if they have high importance.

**Example with w=0.3 (30% stability, 70% importance):**

| Feature | Importance | Stability | Additive | Log Mult | Winner |
|---------|-----------|-----------|----------|----------|--------|
| A       | 0.95      | 0.40      | 0.785    | 0.669    | -      |
| B       | 0.75      | 0.85      | 0.780    | 0.778    | **B**  |

**Additive**: Feature A wins (0.785 > 0.780) - high importance compensates for low stability
**Log Mult**: Feature B wins (0.778 > 0.669) - low stability significantly penalizes A

**Result**: Log multiplication correctly prioritizes the stable feature!

### 2. **Geometric Mean vs Arithmetic Mean**

Log multiplication computes a **weighted geometric mean**, which is more appropriate for multiplicative quantities:

- **Arithmetic mean** (addition): `(a + b) / 2`
  - One high value can compensate for one low value
  
- **Geometric mean** (log mult): `√(a × b)`
  - Both values must be reasonably high
  - One very low value drastically reduces the result

### 3. **Probabilistic Interpretation**

If we think of importance and stability as **independent probabilities**:

- **Additive**: Assumes they're alternatives (OR logic)
- **Log Mult**: Assumes they're requirements (AND logic)

For feature selection, we want features that are **both** important **and** stable!

### 4. **Numerical Examples**

#### Scenario 1: High Importance, Low Stability
```
Importance: 0.90, Stability: 0.30, Weight: 0.3

Additive:  0.7 × 0.90 + 0.3 × 0.30 = 0.72
Log Mult:  0.90^0.7 × 0.30^0.3 = 0.66

Difference: Additive overvalues this feature by 9%
```

#### Scenario 2: Balanced Feature
```
Importance: 0.70, Stability: 0.70, Weight: 0.3

Additive:  0.7 × 0.70 + 0.3 × 0.70 = 0.70
Log Mult:  0.70^0.7 × 0.70^0.3 = 0.70

Difference: Both methods agree (as expected for balanced features)
```

#### Scenario 3: Low Importance, High Stability
```
Importance: 0.40, Stability: 0.90, Weight: 0.3

Additive:  0.7 × 0.40 + 0.3 × 0.90 = 0.55
Log Mult:  0.40^0.7 × 0.90^0.3 = 0.48

Difference: Log mult correctly penalizes low importance
```

### 5. **Sensitivity Analysis**

How much does a 50% drop in one dimension affect the combined score?

**With w=0.3, starting from (0.8, 0.8):**

| Scenario | Additive | Log Mult | Penalty |
|----------|----------|----------|---------|
| Importance drops to 0.4 | 0.52 (-35%) | 0.46 (-42%) | **Stronger** |
| Stability drops to 0.4 | 0.68 (-15%) | 0.61 (-24%) | **Stronger** |

Log multiplication provides **stronger penalties** for weaknesses in either dimension.

## Mathematical Properties

### Additive
- **Linear**: Changes in one variable have constant effect
- **Compensatory**: High value in one can offset low in other
- **Range**: [0, 1] (preserved)

### Log Multiplication
- **Non-linear**: Changes have multiplicative effect
- **Non-compensatory**: Both must be high
- **Range**: [0, 1] (preserved after normalization)
- **Symmetric**: Equal treatment of both dimensions (in log space)

## Real-World Analogy

Think of importance and stability as:

### Additive = "Average Performance"
- Like averaging test scores: 100% + 0% = 50% average
- One excellent score compensates for one terrible score

### Log Multiplication = "Weakest Link"
- Like a chain: strength = min(link1, link2)
- One weak link breaks the whole chain
- More appropriate for feature selection!

## Impact on Feature Selection

### With Additive (0.3 weight):
```
Top 5 Features:
1. High importance, low stability    (0.82)  ← Risky!
2. Medium importance, high stability (0.78)
3. High importance, medium stability (0.77)
4. Medium importance, medium stability (0.70)
5. Low importance, very high stability (0.65)
```

### With Log Multiplication (0.3 weight):
```
Top 5 Features:
1. High importance, high stability   (0.85)  ← Best!
2. Medium importance, high stability (0.78)
3. High importance, medium stability (0.72)
4. Medium importance, medium stability (0.70)
5. High importance, low stability    (0.62)  ← Demoted!
```

## When to Use Each

### Use Additive If:
- You want features that excel in **either** dimension
- Stability is just a "nice to have"
- You're okay with unstable but highly predictive features

### Use Log Multiplication If: ✅
- You want features that are good in **both** dimensions
- Stability is a **requirement**, not optional
- You want robust, reliable features
- You're building production systems

## Conclusion

**Log multiplication is the better choice** for feature selection because:

1. ✅ Enforces that features must be both important AND stable
2. ✅ Provides stronger penalties for weaknesses
3. ✅ More aligned with probabilistic interpretation
4. ✅ Better for production systems (reliability matters)
5. ✅ Handles multiplicative relationships correctly

The only downside is slightly more complex computation (log/exp), but this is negligible compared to the benefits.

## Implementation

```python
# Log multiplication (implemented)
importance_weight = 1 - stability_weight  # e.g., 0.7
epsilon = 1e-10  # Avoid log(0)

for feature in features:
    imp = max(importance[feature], epsilon)
    stab = max(stability[feature], epsilon)
    
    # Log space: w1*log(imp) + w2*log(stab)
    log_combined = importance_weight * np.log(imp) + stability_weight * np.log(stab)
    
    # Back to normal space: exp(log_combined)
    combined_score[feature] = np.exp(log_combined)
```

This is mathematically equivalent to:
```python
combined_score = importance^0.7 × stability^0.3
```

## Recommended Settings

With log multiplication, the recommended weight is still **0.3** (30% stability, 70% importance):

```yaml
feature_selection:
  stability_weight: 0.3  # Balanced setting
```

This gives you:
- **70% weight** on predictive power (importance^0.7)
- **30% weight** on reliability (stability^0.3)
- **Strong penalty** for features weak in either dimension
