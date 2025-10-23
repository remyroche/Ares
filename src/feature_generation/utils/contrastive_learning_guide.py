"""
import warnings
Contrastive Learning Generator Usage Guide

This document provides comprehensive guidance on when, why, and how to use
the ContrastiveLearningGenerator for enhanced representation learning.

CONTRASTIVE LEARNING OVERVIEW
============================

Contrastive learning is a self-supervised learning technique that learns
representations by contrasting positive pairs (similar examples) against
negative pairs (dissimilar examples). This is particularly powerful for
financial time series data where:

1. Market regimes create natural positive/negative pairs
2. Temporal coherence provides supervision signals
3. Cross-timeframe relationships offer rich contrastive signals

WHEN TO USE CONTRASTIVE LEARNING
===============================

✅ **OPTIMAL USE CASES:**

1. **Regime Detection & Classification**
   - Learning representations that distinguish between trending vs ranging markets
   - Identifying volatile vs calm periods
   - Detecting structural breaks in market behavior

2. **Multi-Timeframe Consistency**
   - Learning representations that are consistent across different timeframes
   - Understanding how short-term patterns relate to long-term trends
   - Cross-timeframe feature alignment

3. **Anomaly Detection**
   - Learning normal market patterns to detect unusual behavior
   - Identifying regime shifts and structural changes
   - Market stress detection

4. **Feature Learning for Downstream Tasks**
   - Pre-training representations for regime classification
   - Learning embeddings for similarity-based trading strategies
   - Enhancing existing ML models with learned representations

5. **Temporal Pattern Recognition**
   - Learning recurring market patterns and cycles
   - Understanding seasonal and cyclical behaviors
   - Pattern-based trading signal generation

❌ **SUBOPTIMAL USE CASES:**

1. **Real-time Feature Extraction**
   - Contrastive learning requires batch processing and is computationally expensive
   - Not suitable for high-frequency trading or real-time predictions

2. **Small Datasets**
   - Requires sufficient data to create meaningful positive/negative pairs
   - Minimum 1000+ samples recommended for effective learning

3. **Simple Linear Relationships**
   - Overkill for straightforward price/volume relationships
   - Better alternatives exist for simple momentum/mean-reversion strategies

HOW TO USE CONTRASTIVE LEARNING GENERATOR
========================================

BASIC USAGE
-----------

```python
from src.feature_generation.categories.representation_learning import ContrastiveLearningGenerator

# Initialize with default parameters
generator = ContrastiveLearningGenerator(
    embedding_dim=64,    # Size of learned representations
    temperature=0.1      # Temperature for softmax (lower = sharper contrasts)
)

# Generate features
features_df = generator.generate_feature(market_data)
representations = features_df['contrastive_repr_64_0.1'].values
```

ADVANCED CONFIGURATION
---------------------

```python
# Custom configuration for specific use cases
generator = ContrastiveLearningGenerator(
    embedding_dim=128,           # Larger embeddings for complex patterns
    temperature=0.05,            # Sharper contrasts for regime detection
    # Add custom positive/negative pair generation
    # Add domain-specific contrastive losses
)
```

INTEGRATION WITH EXISTING PIPELINE
---------------------------------

```python
from src.feature_generation.enhanced_feature_engineering_integration import EnhancedFeatureEngineer

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

# Use within enhanced feature engineering pipeline
engineer = EnhancedFeatureEngineer()
features = await engineer.generate_comprehensive_features(
    data_dict,
    include_categories=['representation']  # Include contrastive learning
)
```

BEST PRACTICES
==============

1. **Data Preparation**
   - Ensure sufficient history (1000+ samples minimum)
   - Clean data preprocessing (handle missing values, outliers)
   - Normalize input features before contrastive learning

2. **Hyperparameter Tuning**
   - Start with embedding_dim=64, temperature=0.1
   - Adjust temperature based on contrast sharpness needed
   - Larger embeddings (128-256) for complex market patterns

3. **Training Strategy**
   - Use warm-up periods for stable learning
   - Implement early stopping based on validation loss
   - Monitor for mode collapse (all representations becoming similar)

4. **Evaluation**
   - Use downstream task performance as primary metric
   - Monitor representation diversity and quality
   - Compare against baseline representations

5. **Production Deployment**
   - Cache learned representations for inference speed
   - Implement incremental learning for concept drift
   - Monitor representation quality over time

ADVANCED TECHNIQUES
==================

1. **Multi-View Contrastive Learning**
   - Use different timeframes as different views
   - Apply augmentations (time shifting, noise injection)
   - Learn from multiple perspectives of same market state

2. **Temporal Contrastive Learning**
   - Use time-shifted sequences as positive pairs
   - Learn temporal coherence and patterns
   - Capture market dynamics and transitions

3. **Cross-Asset Contrastive Learning**
   - Learn relationships between correlated assets
   - Use lead-lag relationships as supervision
   - Capture inter-market dependencies

4. **Hierarchical Contrastive Learning**
   - Learn representations at multiple granularities
   - Combine local (short-term) and global (long-term) patterns
   - Multi-scale market understanding

TROUBLESHOOTING
===============

**Problem:** Poor representation quality
**Solution:**
- Increase embedding dimension
- Adjust temperature parameter
- Ensure sufficient positive/negative pairs
- Check data preprocessing quality

**Problem:** Mode collapse (all representations similar)
**Solution:**
- Increase temperature
- Add more diverse negative samples
- Use harder negative mining
- Implement regularization techniques

**Problem:** Slow training convergence
**Solution:**
- Use learning rate scheduling
- Implement gradient clipping
- Increase batch size
- Add momentum to optimizer

**Problem:** Overfitting to training data
**Solution:**
- Use stronger data augmentation
- Implement dropout in representation layers
- Add regularization losses
- Use validation-based early stopping

PERFORMANCE CONSIDERATIONS
=========================

- **Computational Cost:** High (requires matrix operations and backpropagation)
- **Memory Usage:** Moderate to high (stores representations and gradients)
- **Training Time:** 10-60 minutes for typical datasets
- **Inference Speed:** Fast (forward pass only)

COMPARISON WITH ALTERNATIVES
===========================

**vs. Autoencoder:**
- Contrastive learning: Better for capturing relationships and similarities
- Autoencoder: Better for dimensionality reduction and reconstruction

**vs. Supervised Learning:**
- Contrastive learning: No labels required, learns inherent structure
- Supervised learning: Requires labeled data, task-specific optimization

**vs. PCA:**
- Contrastive learning: Non-linear, learns complex patterns
- PCA: Linear, fast, interpretable but limited expressiveness

CONCLUSION
==========

ContrastiveLearningGenerator is a powerful tool for learning rich,
self-supervised representations of market data. Use it when:

- You need to capture complex market relationships
- You have sufficient data and computational resources
- You want representations that work across multiple downstream tasks
- You need to learn from unlabeled market structure

Start with default parameters and gradually customize based on your
specific use case and performance requirements.
"""
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
