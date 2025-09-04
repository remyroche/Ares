from typing import Dict, List, Optional, Union, Any, Tuple
"""
PnL-Aware Loss Function for Keras Models.

This module contains the factory function for creating custom Keras loss functions
that combine standard classification loss with financial components.
"""
from keras import backend as K

def create_pnl_aware_loss(pnl_multiplier: Any=0.1, liquidation_penalty: Any=2.0, reward_boost: Any=1.5) -> Any:
    """
    This is a factory function that creates a custom Keras loss function.
    It combines standard classification loss (cross-entropy) with a financial
    component that heavily penalizes high-risk errors and rewards high-profit
    correct predictions, teaching the model to prioritize capital preservation.
    """

    def pnl_aware_loss(y_true: Any, y_pred: Any) -> None:
        """
        Calculates the combined loss.

        Args:
            y_true: Ground truth tensor with shape (batch_size, num_classes + 2).
                    It contains [one_hot_label, reward_potential, risk_potential].
            y_pred: Predicted probabilities with shape (batch_size, num_classes).
        """
        y_true_labels = y_true[:, :-2]
        reward_potential = y_true[:, -2]
        risk_potential = y_true[:, -1]
        ce_loss = K.categorical_crossentropy(y_true_labels, y_pred)
        true_class_probs = K.sum(y_true_labels * y_pred, axis=-1)
        risk_adjusted_loss = (1 - true_class_probs) * K.exp(risk_potential * liquidation_penalty)
        reward_adjusted_loss = -true_class_probs * reward_potential * reward_boost
        pnl_loss = risk_adjusted_loss + reward_adjusted_loss
        combined_loss = ce_loss + pnl_multiplier * pnl_loss
        return combined_loss
    return pnl_aware_loss