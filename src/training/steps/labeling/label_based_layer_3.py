"""
Layer 3: Multi-Geometry Meta-Models (ORF Implementation)

This file serves as a wrapper for the Orthogonal Random Forest (ORF) Layer 3 implementation.
"""

from typing import List, Tuple, Optional, Any, Dict
import pandas as pd
import numpy as np
import logging

# Import modular Layer 3 implementation
try:
    from src.training.steps.labeling.layer3 import layer3_analyst_lgbm as layer3_analyst_orf
    print("✅ Using ORF-based Layer 3 implementation")
except ImportError as e:
    print(f"⚠️ Failed to import modular Layer 3: {e}")
    layer3_analyst_orf = None

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
    Layer 3: Multi-Geometry Meta-Models (ORF Wrapper)
    
    Delegates to the modular ORF implementation in layer3/core.py.
    """
    if layer3_analyst_orf is not None:
        return layer3_analyst_orf(
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
        raise ImportError("❌ Layer 3 modular implementation not available!")

# Export the main function
__all__ = ['layer3_analyst_lgbm']
