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
    from src.training.steps.labeling.layer3.checkpoint_aware_layer3 import layer3_analyst_lgbm_checkpoint_aware as layer3_analyst_orf
    print("✅ Using checkpoint-aware Layer 3 implementation")
except ImportError as e:
    print(f"⚠️ Failed to import checkpoint-aware Layer 3: {e}")
    # Fallback to non-checkpoint version
    try:
        from src.training.steps.labeling.layer3 import layer3_analyst_lgbm as layer3_analyst_orf
        print("✅ Using standard Layer 3 implementation")
    except ImportError as e2:
        print(f"⚠️ Failed to import standard Layer 3: {e2}")
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
    symbol: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    override_step: Optional[str] = None,
    force_restart: bool = False,
    keep_earlier_checkpoints: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Layer 3: Multi-Geometry Meta-Models (ORF Wrapper)
    
    Delegates to the modular ORF implementation in layer3/core.py.
    Supports checkpoint override functionality.
    
    Args:
        symbol: Trading symbol (required for checkpoint management)
        checkpoint_dir: Optional custom checkpoint directory
        override_step: Step to override from (e.g., 'dual_head_training')
        force_restart: Force restart from beginning (ignores all checkpoints)
        keep_earlier_checkpoints: Keep checkpoints before override step
        ... (other args passed to underlying implementation)
        
    Returns:
        Tuple of (enhanced DataFrame, models dictionary)
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
            config=config,
            symbol=symbol,
            checkpoint_dir=checkpoint_dir,
            override_step=override_step,
            force_restart=force_restart,
            keep_earlier_checkpoints=keep_earlier_checkpoints
        )
    else:
        raise ImportError("❌ Layer 3 modular implementation not available!")

# Export the main function
__all__ = ['layer3_analyst_lgbm']
