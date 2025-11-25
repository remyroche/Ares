"""
Integration Module for Regime Probability Aggregation

This module provides helper functions to integrate regime probability aggregation
into the ml_risk_regime_step. Add these methods to the MLRiskRegimeStep class.

Usage:
    1. Extract regime probabilities from HMM model
    2. Add probability columns to DataFrame
    3. Fit aggregator on training data
    4. Transform probabilities to single scalar
    5. Apply isotonic calibration
    6. Save aggregated score alongside regime columns
"""
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.utils.regime_probability_aggregator import (
    RegimeProbabilityAggregator,
    AggregationMethod,
    compare_aggregation_methods
)


logger = logging.getLogger(__name__)


def extract_regime_probabilities(
    hmm_model: GaussianHMM,
    risk_features: pd.DataFrame,
    valid_mask: np.ndarray,
    n_regimes: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract regime probabilities from trained HMM model.

    Args:
        hmm_model: Trained GaussianHMM model
        risk_features: Risk features DataFrame
        valid_mask: Boolean mask for valid rows (no NaNs)
        n_regimes: Number of regimes

    Returns:
        (prob_df, prob_array)
        - prob_df: DataFrame with regime probability columns
        - prob_array: Numpy array of shape (n_samples, n_regimes)
    """
    logger.info("Extracting regime probabilities from HMM model...")

    # Get clean features
    risk_features_clean = risk_features[valid_mask].copy()
    hmm_features = risk_features_clean.values

    # Compute state probabilities using forward algorithm
    # This gives P(state | observations)
    regime_probs_clean = hmm_model.predict_proba(
        hmm_features,
        lengths=[len(hmm_features)]
    )

    # Create full-size probability array (with NaNs for invalid rows)
    prob_array = np.full((len(risk_features), n_regimes), np.nan)
    prob_array[valid_mask] = regime_probs_clean

    # Create DataFrame with probability columns
    prob_columns = {
        f'risk_regime_{i}_prob': prob_array[:, i]
        for i in range(n_regimes)
    }

    prob_df = pd.DataFrame(prob_columns, index=risk_features.index)

    logger.info(
        f"✅ Extracted regime probabilities: "
        f"{regime_probs_clean.shape[0]} samples, {n_regimes} regimes"
    )

    return prob_df, prob_array


def fit_regime_aggregator(
    regime_probs: np.ndarray,
    forward_returns: np.ndarray,
    regime_stats: Dict[int, Dict[str, float]],
    method: AggregationMethod = AggregationMethod.EXPECTED_RETURN,
    enable_comparison: bool = False
) -> RegimeProbabilityAggregator:
    """
    Fit regime probability aggregator.

    Args:
        regime_probs: Array of shape (n_samples, n_regimes)
        forward_returns: Array of shape (n_samples,)
        regime_stats: Per-regime statistics (from forward metrics)
        method: Aggregation method to use
        enable_comparison: If True, compare all methods and select best

    Returns:
        Fitted aggregator
    """
    logger.info("=" * 80)
    logger.info("🎯 FITTING REGIME PROBABILITY AGGREGATOR")
    logger.info("=" * 80)

    # Remove NaN values
    valid_mask = np.isfinite(regime_probs).all(axis=1) & np.isfinite(forward_returns)
    regime_probs_clean = regime_probs[valid_mask]
    forward_returns_clean = forward_returns[valid_mask]

    logger.info(
        f"Valid samples for aggregator: "
        f"{len(regime_probs_clean)} / {len(regime_probs)}"
    )

    # Compare methods if requested
    if enable_comparison:
        logger.info("\n📊 Comparing aggregation methods...")
        comparison_df = compare_aggregation_methods(
            regime_probs_clean,
            forward_returns_clean,
            regime_stats
        )

        # Select best method by ROC-AUC
        best_method_name = comparison_df.index[0]
        method = AggregationMethod(best_method_name)
        logger.info(f"\n✅ Selected best method: {method.value}")

    # Fit aggregator with selected method
    aggregator = RegimeProbabilityAggregator(
        method=method,
        isotonic_calibration=True
    )

    aggregator.fit(
        regime_probs_clean,
        forward_returns_clean,
        regime_stats
    )

    # Evaluate
    metrics = aggregator.evaluate(regime_probs_clean, forward_returns_clean)

    logger.info("\n📈 Aggregator Performance:")
    logger.info(f"  Method: {method.value}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
    logger.info(f"  Spearman: {metrics['spearman']:.4f}")
    logger.info("=" * 80)

    return aggregator


def add_aggregated_score(
    prob_df: pd.DataFrame,
    aggregator: RegimeProbabilityAggregator,
    score_column_name: str = 'risk_regime_score'
) -> pd.DataFrame:
    """
    Add aggregated regime score to probability DataFrame.

    Args:
        prob_df: DataFrame with regime probability columns
        aggregator: Fitted aggregator
        score_column_name: Name for aggregated score column

    Returns:
        DataFrame with added score column
    """
    logger.info(f"Adding aggregated regime score: {score_column_name}")

    # Extract probability array from DataFrame
    prob_cols = [c for c in prob_df.columns if c.startswith('risk_regime_') and c.endswith('_prob')]
    prob_array = prob_df[prob_cols].values

    # Transform to single scalar
    valid_mask = np.isfinite(prob_array).all(axis=1)
    scores = np.full(len(prob_df), np.nan)

    if valid_mask.any():
        scores[valid_mask] = aggregator.transform(prob_array[valid_mask])

    # Add to DataFrame
    prob_df[score_column_name] = scores

    logger.info(
        f"✅ Added {score_column_name}: "
        f"{np.sum(np.isfinite(scores))} / {len(scores)} valid values"
    )
    logger.info(
        f"  Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]"
    )
    logger.info(
        f"  Score mean: {np.nanmean(scores):.4f} ± {np.nanstd(scores):.4f}"
    )

    return prob_df


def save_aggregator_model(
    aggregator: RegimeProbabilityAggregator,
    output_path: str
):
    """
    Save fitted aggregator to disk.

    Args:
        aggregator: Fitted aggregator
        output_path: Path to save model
    """
    aggregator.save(output_path)
    logger.info(f"💾 Saved aggregator model to: {output_path}")


# ============================================================================
# Integration Example for MLRiskRegimeStep
# ============================================================================

def integrate_aggregation_into_step(
    hmm_model: GaussianHMM,
    risk_df: pd.DataFrame,
    risk_features: pd.DataFrame,
    valid_mask: np.ndarray,
    regime_labels: np.ndarray,
    forward_metrics: Dict[str, Any],
    n_regimes: int,
    aggregation_method: str = "expected_return",
    enable_method_comparison: bool = False
) -> Tuple[pd.DataFrame, RegimeProbabilityAggregator]:
    """
    Full integration workflow for regime probability aggregation.

    This function demonstrates how to integrate aggregation into
    the ml_risk_regime_step's run() method.

    Args:
        hmm_model: Trained HMM model
        risk_df: Risk features DataFrame (with 'close' prices)
        risk_features: Features used for HMM training
        valid_mask: Boolean mask for valid rows
        regime_labels: Hard regime labels
        forward_metrics: Forward return metrics (from _calculate_forward_returns_and_sharpe)
        n_regimes: Number of regimes
        aggregation_method: Aggregation method name
        enable_method_comparison: Compare all methods and select best

    Returns:
        (updated_risk_df, fitted_aggregator)
    """
    logger.info("\n" + "=" * 80)
    logger.info("🎯 REGIME PROBABILITY AGGREGATION INTEGRATION")
    logger.info("=" * 80 + "\n")

    # Step 1: Extract regime probabilities
    prob_df, prob_array = extract_regime_probabilities(
        hmm_model,
        risk_features,
        valid_mask,
        n_regimes
    )

    # Step 2: Add probability columns to risk_df
    for col in prob_df.columns:
        risk_df[col] = prob_df[col]

    logger.info(f"✅ Added {len(prob_df.columns)} probability columns to risk_df")

    # Step 3: Calculate forward returns for aggregator fitting
    if 'close' not in risk_df.columns:
        logger.warning("No 'close' column found. Skipping aggregation.")
        return risk_df, None

    close_prices = risk_df['close'].values
    horizon = 4  # 4h forward for 1h data

    forward_returns = np.full(len(close_prices), np.nan)
    for i in range(len(close_prices) - horizon):
        forward_returns[i] = np.log(close_prices[i + horizon] / close_prices[i])

    # Step 4: Extract regime statistics for weighting
    # Use forward_metrics from the step (e.g., forward_metrics['4h'])
    regime_stats = forward_metrics.get('4h', {})

    if not regime_stats:
        logger.warning("No forward metrics found. Computing from data.")
        # Fallback: will be computed by aggregator
        regime_stats = None

    # Step 5: Fit aggregator
    try:
        method_enum = AggregationMethod(aggregation_method)
    except ValueError:
        logger.warning(
            f"Unknown method '{aggregation_method}', defaulting to 'expected_return'"
        )
        method_enum = AggregationMethod.EXPECTED_RETURN

    aggregator = fit_regime_aggregator(
        regime_probs=prob_array,
        forward_returns=forward_returns,
        regime_stats=regime_stats,
        method=method_enum,
        enable_comparison=enable_method_comparison
    )

    # Step 6: Add aggregated score
    risk_df = add_aggregated_score(
        prob_df=risk_df,
        aggregator=aggregator,
        score_column_name='risk_regime_score'
    )

    logger.info("\n✅ Regime probability aggregation integration complete!")
    logger.info("=" * 80 + "\n")

    return risk_df, aggregator


# ============================================================================
# Usage Instructions
# ============================================================================

"""
To integrate into ml_risk_regime_step.py, add the following to the run() method
after line 310 (after calculating forward_metrics):

```python
# Import at top of file
from src.training.steps.market_analysis.regime_aggregation_integration import (
    integrate_aggregation_into_step,
    save_aggregator_model
)

# In run() method, after line 310:
# Calculate forward returns and Sharpe ratios
forward_metrics = self._calculate_forward_returns_and_sharpe(
    risk_df, regime_labels, horizons=[4]
)

# === NEW: Integrate regime probability aggregation ===
aggregation_method = str(config.get("aggregation_method", "expected_return"))
enable_comparison = bool(config.get("enable_aggregation_comparison", False))

risk_df, aggregator = integrate_aggregation_into_step(
    hmm_model=hmm_model,
    risk_df=risk_df,
    risk_features=risk_df[available_features],  # From _train_hmm_regimes
    valid_mask=valid_mask,  # From _train_hmm_regimes
    regime_labels=regime_labels,
    forward_metrics=forward_metrics,
    n_regimes=n_regimes,
    aggregation_method=aggregation_method,
    enable_method_comparison=enable_comparison
)

# Save aggregator model alongside HMM model
if aggregator is not None:
    aggregator_path = f"artifacts/{symbol}_{exchange}_{regime_timeframe}_regime_aggregator.pkl"
    save_aggregator_model(aggregator, aggregator_path)
```

Then the risk_df will include:
- risk_regime_0_prob, risk_regime_1_prob, ... (regime probabilities)
- risk_regime_score (aggregated 0-1 scalar with isotonic calibration)

Configuration options (add to config dict):
- aggregation_method: "expected_return", "expected_sharpe", "logistic_regression",
                      "neural_network", or "pca_first"
- enable_aggregation_comparison: True to compare all methods and select best
"""
