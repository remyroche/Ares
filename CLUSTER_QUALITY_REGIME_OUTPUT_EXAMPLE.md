# Cluster Quality Assessor - Enhanced Output Example

## Complete Output Structure with Regime Classification

### Example ClusterQualityMetrics Output

```python
{
    # === EXISTING CORE METRICS (unchanged) ===
    "silhouette_score": 0.42,
    "davies_bouldin_score": 1.35,
    "calinski_harabasz_score": 245.6,
    "within_regime_cv": 0.15,
    "between_regime_cv": 0.68,
    "temporal_smoothness": 0.87,
    "regime_persistence": 45.2,
    "n_regimes": 4,
    "noise_ratio": 0.08,
    "balance_score": 0.73,
    
    # === NEW: REGIME TYPE CLASSIFICATION ===
    "regime_type_per_cluster": {
        0: "trending",
        1: "mean_reverting",
        2: "volatile",
        3: "stable"
    },
    
    # === ENHANCED: PER-REGIME METRICS ===
    "per_regime_metrics": {
        0: {
            # Existing metrics
            "size": 450,
            "percentage": 28.5,
            "mean_return": 0.0012,
            "volatility": 0.018,
            "sharpe": 0.67,
            
            # NEW: Regime type classification
            "regime_type": "trending",
            
            # NEW: Classification scores (all detection metrics)
            "classification_scores": {
                "trend_strength": 0.75,
                "trend_persistence": 0.45,
                "mean_reversion_strength": -0.45,
                "volatility_level": 0.018,
                "volatility_clustering": 0.25,
                "stability_score": 0.62
            },
            
            # NEW: Regime-specific metrics (trending)
            "regime_specific_metrics": {
                "trend_direction": "bullish",
                "trend_consistency": 0.82,
                "trend_acceleration": 0.15
            }
        },
        
        1: {
            "size": 380,
            "percentage": 24.1,
            "mean_return": 0.0005,
            "volatility": 0.012,
            "sharpe": 0.42,
            
            # Mean reverting regime
            "regime_type": "mean_reverting",
            "classification_scores": {
                "trend_strength": 0.25,
                "trend_persistence": -0.35,  # Negative = mean reverting
                "mean_reversion_strength": 0.35,
                "volatility_level": 0.012,
                "volatility_clustering": 0.15,
                "stability_score": 0.55
            },
            "regime_specific_metrics": {
                "reversion_center": 0.0005,
                "reversion_speed": 12.5,
                "reversion_range": 0.008
            }
        },
        
        2: {
            "size": 420,
            "percentage": 26.6,
            "mean_return": -0.0003,
            "volatility": 0.035,
            "sharpe": -0.09,
            
            # Volatile regime
            "regime_type": "volatile",
            "classification_scores": {
                "trend_strength": 0.20,
                "trend_persistence": 0.10,
                "mean_reversion_strength": -0.10,
                "volatility_level": 0.035,
                "volatility_clustering": 0.65,  # High clustering
                "stability_score": 0.30
            },
            "regime_specific_metrics": {
                "volatility_regime": "high",
                "volatility_persistence": 0.58,
                "extreme_move_frequency": 0.12
            }
        },
        
        3: {
            "size": 330,
            "percentage": 20.9,
            "mean_return": 0.0001,
            "volatility": 0.006,
            "sharpe": 0.17,
            
            # Stable regime
            "regime_type": "stable",
            "classification_scores": {
                "trend_strength": 0.15,
                "trend_persistence": 0.05,
                "mean_reversion_strength": -0.05,
                "volatility_level": 0.006,
                "volatility_clustering": 0.10,
                "stability_score": 0.85
            },
            "regime_specific_metrics": {
                "stability_regime": "low_volatility",
                "mean_return": 0.0001,
                "volatility": 0.006,
                "stability_coefficient": 0.85
            }
        }
    },
    
    # === NEW: ECONOMIC INTERPRETATION ===
    "economic_interpretation": {
        
        # Regime summary
        "regime_summary": {
            "total_regimes": 4,
            "regime_type_distribution": {
                "trending": 1,
                "mean_reverting": 1,
                "volatile": 1,
                "stable": 1
            },
            "dominant_regime": "trending"  # Based on size or occurrence
        },
        
        # Performance comparison by regime type
        "performance_comparison": {
            "trending": {
                "avg_return": 0.0012,
                "avg_volatility": 0.018,
                "avg_sharpe": 0.67,
                "num_regimes": 1,
                "regime_ids": [0]
            },
            "mean_reverting": {
                "avg_return": 0.0005,
                "avg_volatility": 0.012,
                "avg_sharpe": 0.42,
                "num_regimes": 1,
                "regime_ids": [1]
            },
            "volatile": {
                "avg_return": -0.0003,
                "avg_volatility": 0.035,
                "avg_sharpe": -0.09,
                "num_regimes": 1,
                "regime_ids": [2]
            },
            "stable": {
                "avg_return": 0.0001,
                "avg_volatility": 0.006,
                "avg_sharpe": 0.17,
                "num_regimes": 1,
                "regime_ids": [3]
            }
        },
        
        # Trading implications
        "trading_implications": {
            
            # Best regime to trade
            "most_profitable_regime": {
                "regime_id": 0,
                "regime_type": "trending",
                "sharpe_ratio": 0.67,
                "mean_return": 0.0012,
                "volatility": 0.018,
                "characteristics": {
                    "trend_direction": "bullish",
                    "trend_consistency": 0.82,
                    "trend_acceleration": 0.15
                }
            },
            
            # Worst regime to avoid
            "least_profitable_regime": {
                "regime_id": 2,
                "regime_type": "volatile",
                "sharpe_ratio": -0.09,
                "mean_return": -0.0003,
                "volatility": 0.035,
                "characteristics": {
                    "volatility_regime": "high",
                    "volatility_persistence": 0.58,
                    "extreme_move_frequency": 0.12
                }
            },
            
            # Strategy recommendations
            "strategy_recommendations": [
                {
                    "strategy": "trend_following",
                    "target_regime": 0,
                    "expected_sharpe": 0.67,
                    "confidence": 0.45  # trend_persistence score
                },
                {
                    "strategy": "risk_avoidance",
                    "target_regimes": [2],
                    "rationale": "high drawdown or negative sharpe"
                }
            ]
        },
        
        # Risk characteristics per regime
        "risk_characteristics": {
            "regime_0": {
                "regime_id": 0,
                "volatility": 0.018,
                "max_drawdown": -0.08,
                "skewness": 0.15,
                "trend_consistency": 0.82,
                "trend_direction": "bullish"
            },
            "regime_1": {
                "regime_id": 1,
                "volatility": 0.012,
                "max_drawdown": -0.05,
                "skewness": -0.05,
                "reversion_speed": 12.5,
                "reversion_range": 0.008
            },
            "regime_2": {
                "regime_id": 2,
                "volatility": 0.035,
                "max_drawdown": -0.18,
                "skewness": -0.35,
                "extreme_move_frequency": 0.12,
                "volatility_persistence": 0.58
            },
            "regime_3": {
                "regime_id": 3,
                "volatility": 0.006,
                "max_drawdown": -0.02,
                "skewness": 0.05
            }
        },
        
        # Regime stability
        "regime_transitions": {
            "balance": {
                "most_common_regime_pct": 28.5,
                "least_common_regime_pct": 20.9,
                "size_distribution_std": 48.2
            }
        }
    },
    
    # === EXISTING METRICS (unchanged) ===
    "quality_score": 0.72,
    "predictive_power": 0.58,
    "timestamp": "2025-10-28T12:34:56"
}
```

## Key Features of Enhanced Output

### 1. Regime Type Classification
Every cluster is labeled with its economic regime type:
- **TRENDING:** Strong directional movement with persistence
- **MEAN_REVERTING:** Prices oscillate around a center
- **VOLATILE:** High volatility with clustering
- **STABLE:** Low volatility, range-bound

### 2. Classification Scores
For each regime, see ALL detection metrics used to classify it:
```python
"classification_scores": {
    "trend_strength": 0.75,           # How strong is the trend?
    "trend_persistence": 0.45,        # Does the trend continue?
    "mean_reversion_strength": -0.45, # Does it revert to mean?
    "volatility_level": 0.018,        # How volatile?
    "volatility_clustering": 0.25,    # Does volatility cluster?
    "stability_score": 0.62           # How stable overall?
}
```

### 3. Regime-Specific Metrics
Different metrics for different regime types:

**Trending Regimes:**
- Direction (bullish/bearish)
- Consistency (% of returns in trend direction)
- Acceleration (is trend speeding up?)

**Mean Reverting Regimes:**
- Reversion center (equilibrium level)
- Reversion speed (how fast it reverts)
- Reversion range (typical deviation)

**Volatile Regimes:**
- Extreme move frequency
- Volatility persistence

**Stable Regimes:**
- Stability coefficient

### 4. Economic Interpretation

**Performance Comparison:**
Compare average returns, volatility, and Sharpe across regime types

**Trading Implications:**
- Which regime is most profitable?
- Which regime to avoid?
- Strategy recommendations with expected Sharpe

**Risk Characteristics:**
Per-regime risk profile with regime-specific risk metrics

## Example Usage in Code

```python
# After running quality assessment
metrics = assessor.assess_quality(...)

# Quick regime type lookup
print("Regime Types:")
for regime_id, regime_type in metrics.regime_type_per_cluster.items():
    print(f"  Regime {regime_id}: {regime_type}")

# Get best regime to trade
best = metrics.economic_interpretation['trading_implications']['most_profitable_regime']
print(f"\nBest regime to trade: {best['regime_id']} ({best['regime_type']})")
print(f"Expected Sharpe: {best['sharpe_ratio']:.2f}")
print(f"Characteristics: {best['characteristics']}")

# Get strategy recommendations
for rec in metrics.economic_interpretation['trading_implications']['strategy_recommendations']:
    print(f"\nStrategy: {rec['strategy']}")
    print(f"Target: Regime {rec.get('target_regime', rec.get('target_regimes'))}")
    if 'expected_sharpe' in rec:
        print(f"Expected Sharpe: {rec['expected_sharpe']:.2f}")

# Analyze specific regime
regime_id = 0
regime = metrics.per_regime_metrics[regime_id]
print(f"\nRegime {regime_id} Analysis:")
print(f"Type: {regime['regime_type']}")
print(f"Size: {regime['size']} samples ({regime['percentage']:.1f}%)")
print(f"Classification Scores:")
for metric, score in regime['classification_scores'].items():
    print(f"  {metric}: {score:.3f}")
print(f"Specific Metrics:")
for metric, value in regime['regime_specific_metrics'].items():
    print(f"  {metric}: {value}")
```

## Actionable Insights

The enhanced output provides answers to:

1. **"What type of market regime is this cluster?"**
   → Check `regime_type`

2. **"Why was it classified as that type?"**
   → Check `classification_scores`

3. **"What's special about this regime?"**
   → Check `regime_specific_metrics`

4. **"Which regime should I trade?"**
   → Check `trading_implications.most_profitable_regime`

5. **"What strategy should I use in this regime?"**
   → Check `trading_implications.strategy_recommendations`

6. **"What are the risks in this regime?"**
   → Check `risk_characteristics[regime_id]`

7. **"How do different regime types perform?"**
   → Check `performance_comparison`
