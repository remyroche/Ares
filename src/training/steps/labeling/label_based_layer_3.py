"""
Layer 3: Multi-Geometry Meta-Models (Clean Wrapper)

This file serves as a clean wrapper for the modular Layer 3 implementation.
All main functionality has been moved to the modular layer3/ directory.
"""

from typing import List, Tuple, Optional, Any, Dict
import pandas as pd
import numpy as np
import logging

# Import modular Layer 3 implementation
try:
    from src.training.steps.labeling.layer3 import layer3_analyst_lgbm as layer3_analyst_lgbm_modular
    print("✅ Using modular Layer 3 implementation")
except ImportError as e:
    print(f"⚠️ Failed to import modular Layer 3: {e}")
    # Fallback implementation would go here
    layer3_analyst_lgbm_modular = None

# Import essential utilities that might be called by other parts
from src.feature_generation.categories.layer3_specific_features import generate_layer3_features

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Layer 3: Multi-Geometry Meta-Models

    Performance Optimization Options (add to config dict):
    - fast_mode: True/False (default: False) - Skip expensive operations for 3-5x speedup
    - skip_layer0_integration: True/False (default: False) - Skip Layer 0 feature engineering
    - skip_feature_selection: True/False (default: False) - Skip hierarchical feature filtering
    - cv_folds: int - Number of CV folds (default 5, fast_mode uses 3)
    - n_geometries: int - Number of geometries to generate (default 4, fast_mode uses 2)
    - top_k_geometries: int - Number of geometries to select (default 4, fast_mode uses 2)

    Note: All optimization options default to False for full quality processing.
    Set fast_mode: True for 3-5x speedup during development/iteration.

    Example fast config: {'fast_mode': True}
    """
    """
    Layer 3: Multi-Geometry Meta-Models (Clean Wrapper)
    
    This is a clean wrapper that delegates to the modular Layer 3 implementation.
    All main functionality has been moved to the modular layer3/ directory.
    
    Args:
        oof_df: Out-of-sample dataframe with base model predictions
        base_model_cols: List of base model columns
        target_col: Target column name
        train_split_date: Optional train split date
        sample_weight: Optional sample weights
        layer1_weight: Layer 1 sample weights
        layer2_weight: Layer 2 sample weights
        layer2_weight_quality: Layer 2 quality weights
        net_returns: Net returns series
        market_data: Market data (OHLCV)
        config: Configuration dictionary
        
    Returns:
        Tuple of (enhanced dataframe, models dictionary)
    """
    if layer3_analyst_lgbm_modular is not None:
        # Use modular implementation
        return layer3_analyst_lgbm_modular(
            oof_df=oof_df,
            base_model_cols=base_model_cols,
            target_col=target_col,
            train_split_date=train_split_date,
            sample_weight=sample_weight,
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            layer2_weight_quality=layer2_weight_quality,
            net_returns=net_returns,
            market_data=market_data,
            config=config
        )
    else:
        # Fallback implementation (simplified)
        print("⚠️ Using fallback Layer 3 implementation")
        return _fallback_layer3_implementation(
            oof_df, base_model_cols, target_col, train_split_date,
            sample_weight, layer1_weight, layer2_weight, 
            layer2_weight_quality, net_returns, market_data, config
        )

def _fallback_layer3_implementation(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fast fail Layer 3 implementation.
    """
    error_msg = (
        "❌ Layer 3 modular implementation not available!\n\n"
        "Please ensure the modular Layer 3 implementation is properly installed:\n"
        "- Check layer3/ directory exists\n"
        "- Verify all required dependencies are installed\n"
        "- Ensure proper imports are available\n\n"
        "This is a fast fail to prevent silent fallback behavior."
    )
    raise ImportError(error_msg)


def plot_diagnostics(y_true, y_prob, output_path: Optional[str] = None) -> None:
    """
    Legacy diagnostic plotting function (kept for backward compatibility).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, log_loss
    
    try:
        y_true_numeric = pd.to_numeric(y_true, errors="coerce")
        y_prob_numeric = pd.to_numeric(y_prob, errors="coerce")
        mask = ~y_true_numeric.isna() & ~y_prob_numeric.isna()
        y_true = y_true_numeric[mask]
        y_prob = y_prob_numeric[mask]

        if len(y_true) == 0:
            return

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration plot
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax[0].plot(prob_pred, prob_true, marker="o", linewidth=2, label="Meta-Model")
        ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect")
        ax[0].set_xlabel("Predicted Probability")
        ax[0].set_ylabel("Actual Win Rate")
        ax[0].set_title("Calibration (Reliability)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        sns.histplot(y_prob, bins=20, kde=True, ax=ax[1], color="purple", alpha=0.6)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel("Predicted Probability")
        ax[1].set_title("Resolution (Confidence Distribution)")
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close(fig)
    except Exception as exc:
        print(f"Failed to generate plots: {exc}")
    
    return

# Legacy function that might be called by other parts
def generate_layer3_features(df, base_model_cols):
    """
    Legacy Layer 3 feature generation.
    """
    try:
        from src.feature_generation.categories.layer3_specific_features import generate_layer3_features as generate_features
        return generate_features(df, base_model_cols)
    except ImportError:
        print("⚠️ Layer 3 feature generation not available")
        return df

# Export the main function
__all__ = ['layer3_analyst_lgbm', 'plot_diagnostics', 'generate_layer3_features']
