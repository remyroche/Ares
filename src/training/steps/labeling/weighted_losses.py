"""
Weighted Loss Functions for Financial Machine Learning

Implements De Prado's approach to sample-weighted loss functions:
1. Absolute Return Weighting: Weight samples by |r| for PnL focus
2. Enhanced Weighted Log-Loss: Custom loss for big moves emphasis
3. Financial Loss Functions: PnL-oriented loss calculations
4. Asymmetric / Downside Losses: Penalize adverse outcomes more heavily

These losses force models to be accurate on high-impact moves while allowing
flexibility on small, noisy trades.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

from .invariant_risk_minimization import FinancialIRM, IRMLossFunction, apply_irm_training, quick_irm_evaluation

def compute_absolute_return_weights(
    returns: pd.Series,
    min_weight: float = 0.1,
    max_weight: float = 5.0,
    percentile_clip: float = 0.95,
    apply_log_transform: bool = True
) -> np.ndarray:
    """
    Compute absolute return-based sample weights.
    
    Big moves get higher weights, forcing model to focus on high-impact trades.
    
    Args:
        returns: Return series
        min_weight: Minimum weight to prevent underweighting
        max_weight: Maximum weight to prevent overfitting to outliers
        percentile_clip: Clip weights at this percentile to reduce outlier impact
        apply_log_transform: Whether to apply log transform for smoother weighting
        
    Returns:
        Array of sample weights
    """
    # Calculate absolute returns
    abs_returns = np.abs(returns.fillna(0))
    
    # Apply log transform for smoother weighting (optional)
    if apply_log_transform:
        # Add small constant to avoid log(0)
        abs_returns = np.log1p(abs_returns)
    
    # Normalize to create weights
    if abs_returns.sum() == 0:
        weights = np.ones_like(abs_returns)
    else:
        weights = abs_returns / abs_returns.mean()
    
    # Clip extreme weights
    if percentile_clip < 1.0:
        clip_threshold = np.percentile(weights, percentile_clip * 100)
        weights = np.minimum(weights, clip_threshold)
    
    # Apply min/max bounds
    weights = np.clip(weights, min_weight, max_weight)
    
    # Normalize to mean 1.0
    weights = weights / weights.mean()
    
    return weights

def compute_pnl_weighted_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    loss_type: str = 'logloss',
    alpha: float = 0.5,
    epsilon: float = 1e-8
) -> float:
    """
    Compute PnL-weighted loss function.
    
    Combines standard loss with PnL-based weighting for financial focus.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        returns: Return series for weighting
        loss_type: Type of loss ('logloss', 'brier', 'focal')
        alpha: Weighting factor (0=standard, 1=full PnL weighting)
        epsilon: Small constant for numerical stability
        
    Returns:
        Weighted loss value
    """
    # Compute absolute return weights
    abs_ret_weights = compute_absolute_return_weights(pd.Series(returns))
    
    # Normalize weights
    weights = (1 - alpha) + alpha * abs_ret_weights
    
    # Compute base loss
    if loss_type == 'logloss':
        # Clip predictions for numerical stability
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        base_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    elif loss_type == 'brier':
        base_loss = (y_true - y_pred) ** 2
    elif loss_type == 'focal':
        # Simplified focal loss
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        pt = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)
        base_loss = -(1 - pt) ** 2 * np.log(pt)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    # Apply weights
    weighted_loss = weights * base_loss
    
    return np.mean(weighted_loss)

# ===== Asymmetric / Downside Losses =====

class AsymmetricHuberLoss(nn.Module):
    """
    Asymmetric Huber Loss that penalizes adverse errors (wrong direction)
    more heavily than magnitude errors in the correct direction.

    Formula:
    L(y, p) =
      if sign(y) == sign(p):
         0.5 * error^2                  (if |error| < delta)
         delta * (|error| - 0.5*delta)  (if |error| >= delta)
      else (wrong direction):
         penalty_factor * (0.5 * error^2)

    This encourages the model to be 'conservative' about sign changes but
    accurate on magnitude when correct.
    """
    def __init__(self, delta: float = 1.0, penalty_factor: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.penalty_factor = penalty_factor
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_true - y_pred
        abs_error = torch.abs(error)

        # Standard Huber
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        base_loss = torch.where(abs_error < self.delta, quadratic, linear)

        # Asymmetric penalty for sign mismatch (adverse PnL potential)
        # Note: y_true and y_pred are often in [-1, 1] for signals
        # If signs differ, y_true * y_pred < 0
        sign_mismatch = (y_true * y_pred) < 0

        weighted_loss = torch.where(sign_mismatch, base_loss * self.penalty_factor, base_loss)

        if self.reduction == 'mean':
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)
        return weighted_loss

class DownsideMSELoss(nn.Module):
    """
    MSE Loss that penalizes negative P&L implications (Predicted * Actual < 0)
    much more heavily.
    """
    def __init__(self, adverse_weight: float = 5.0, reduction: str = 'mean'):
        super().__init__()
        self.adverse_weight = adverse_weight
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = (y_true - y_pred) ** 2

        # Adverse: We predicted Long (Positive), Actual was Short (Negative) OR vice versa
        # In regression context, we care about the implied PnL: y_pred * y_true
        # If PnL < 0, it's an adverse error.

        # Note: If target is raw return, this works directly.
        # If target is binary label, use standard CrossEntropy/Focal.
        # Assuming y_true is continuous return proxy here.

        pnl_proxy = y_pred * y_true
        weights = torch.ones_like(mse)
        weights[pnl_proxy < 0] = self.adverse_weight

        loss = weights * mse

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss

# ===== Existing Losses =====

class WeightedLogLoss(nn.Module):
    """
    PyTorch implementation of weighted log loss with absolute return weighting.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Initialize weighted log loss.
        
        Args:
            alpha: Weighting factor (0=standard, 1=full absolute return weighting)
            epsilon: Small constant for numerical stability
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            returns: Return values for weighting (optional)
            
        Returns:
            Weighted loss
        """
        # Clip predictions for numerical stability
        y_pred_clipped = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Standard log loss
        log_loss = -(y_true * torch.log(y_pred_clipped) + (1 - y_true) * torch.log(1 - y_pred_clipped))
        
        # Apply absolute return weighting if returns provided
        if returns is not None and self.alpha > 0:
            abs_returns = torch.abs(returns)
            weights = 1.0 + self.alpha * (abs_returns / abs_returns.mean())
            weights = torch.clamp(weights, 0.1, 5.0)  # Reasonable bounds
            log_loss = log_loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(log_loss)
        elif self.reduction == 'sum':
            return torch.sum(log_loss)
        else:
            return log_loss

class FinancialFocalLoss(nn.Module):
    """
    Focal loss designed for financial applications with absolute return weighting.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        return_alpha: float = 0.5,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Initialize financial focal loss.
        
        Args:
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter (focusing parameter)
            return_alpha: Absolute return weighting factor
            epsilon: Small constant for numerical stability
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.return_alpha = return_alpha
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        """
        # Clip predictions
        y_pred_clipped = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Calculate pt
        pt = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)
        
        # Calculate focal loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)
        
        # Apply absolute return weighting
        if returns is not None and self.return_alpha > 0:
            abs_returns = torch.abs(returns)
            weights = 1.0 + self.return_alpha * (abs_returns / abs_returns.mean())
            weights = torch.clamp(weights, 0.1, 5.0)
            focal_loss = focal_loss * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def enhanced_weighted_logloss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    beta: float = 0.3,
    epsilon: float = 1e-8
) -> float:
    """
    Enhanced weighted log loss combining multiple weighting schemes.
    
    Combines:
    1. Standard sample weights (e.g., uniqueness, class weights)
    2. Absolute return weights (big moves focus)
    3. Prediction confidence weights (uncertainty weighting)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        returns: Return series
        sample_weights: Optional sample weights from other sources
        alpha: Absolute return weighting factor
        beta: Confidence weighting factor
        epsilon: Numerical stability constant
        
    Returns:
        Enhanced weighted loss
    """
    # Clip predictions
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Standard log loss
    log_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    # Base weights
    weights = np.ones_like(y_true)
    
    # Apply sample weights if provided
    if sample_weights is not None:
        weights *= sample_weights
    
    # Apply absolute return weights
    if alpha > 0:
        abs_ret_weights = compute_absolute_return_weights(pd.Series(returns))
        weights *= (1.0 + alpha * (abs_ret_weights - 1.0))
    
    # Apply confidence weighting (downweight overconfident predictions)
    if beta > 0:
        confidence = np.maximum(y_pred_clipped, 1 - y_pred_clipped)
        confidence_weights = 1.0 - beta * (confidence - 0.5) * 2  # Scale to [0,1]
        confidence_weights = np.clip(confidence_weights, 0.5, 1.5)
        weights *= confidence_weights
    
    # Apply weights and compute mean
    weighted_loss = weights * log_loss
    
    return np.mean(weighted_loss)

class WeightedLossCalculator:
    """
    Utility class for calculating various weighted losses.
    """
    
    def __init__(
        self,
        return_alpha: float = 0.5,
        confidence_beta: float = 0.3,
        min_weight: float = 0.1,
        max_weight: float = 5.0
    ):
        """
        Initialize weighted loss calculator.
        
        Args:
            return_alpha: Absolute return weighting factor
            confidence_beta: Confidence weighting factor
            min_weight: Minimum sample weight
            max_weight: Maximum sample weight
        """
        self.return_alpha = return_alpha
        self.confidence_beta = confidence_beta
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def compute_weights(
        self,
        returns: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute combined sample weights.
        
        Args:
            returns: Return series
            sample_weights: Optional base sample weights
            predictions: Optional predictions for confidence weighting
            
        Returns:
            Combined weights array
        """
        # Start with base weights
        weights = np.ones_like(returns)
        
        # Apply sample weights
        if sample_weights is not None:
            weights *= sample_weights
        
        # Apply absolute return weights
        if self.return_alpha > 0:
            abs_ret_weights = compute_absolute_return_weights(
                pd.Series(returns),
                min_weight=self.min_weight,
                max_weight=self.max_weight
            )
            weights *= (1.0 + self.return_alpha * (abs_ret_weights - 1.0))
        
        # Apply confidence weights
        if self.confidence_beta > 0 and predictions is not None:
            confidence = np.maximum(predictions, 1 - predictions)
            confidence_weights = 1.0 - self.confidence_beta * (confidence - 0.5) * 2
            confidence_weights = np.clip(confidence_weights, 0.5, 1.5)
            weights *= confidence_weights
        
        # Normalize to mean 1.0
        weights = weights / weights.mean()
        
        return weights
    
    def weighted_logloss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate weighted log loss.
        """
        weights = self.compute_weights(returns, sample_weights, y_pred)
        
        # Clip predictions
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        
        # Calculate weighted log loss
        log_loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        weighted_loss = weights * log_loss
        
        return np.mean(weighted_loss)

# Convenience functions for quick usage
def quick_abs_return_weights(returns: pd.Series) -> np.ndarray:
    """Quick absolute return weighting with default settings."""
    return compute_absolute_return_weights(returns)

def quick_pnl_logloss(y_true: np.ndarray, y_pred: np.ndarray, returns: np.ndarray) -> float:
    """Quick PnL-weighted log loss with default settings."""
    return compute_pnl_weighted_loss(y_true, y_pred, returns, loss_type='logloss')


# ===== IRM (Invariant Risk Minimization) Loss Functions =====

def irm_weighted_logloss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    environment_masks: Dict[str, np.ndarray],
    sample_weights: Optional[np.ndarray] = None,
    lambda_irm: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    epsilon: float = 1e-8
) -> Tuple[float, Dict[str, float]]:
    """
    IRM-enhanced weighted log loss with environment invariance.
    
    Combines standard weighted loss with invariance penalty across environments.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        returns: Return series for absolute return weighting
        environment_masks: Environment masks from custom regime features
        sample_weights: Optional sample weights
        lambda_irm: IRM penalty weight
        alpha: Absolute return weighting factor
        beta: Confidence weighting factor
        epsilon: Numerical stability constant
        
    Returns:
        Tuple of (total_loss, loss_breakdown)
    """
    try:
        from .invariant_risk_minimization import FinancialIRM
        
        # Create IRM instance
        irm = FinancialIRM(lambda_irm=lambda_irm, verbose=False)
        
        # Compute combined weights
        weights = np.ones_like(y_true)
        
        # Apply sample weights
        if sample_weights is not None:
            weights *= sample_weights
        
        # Apply absolute return weights
        if alpha > 0:
            abs_ret_weights = compute_absolute_return_weights(pd.Series(returns))
            weights *= (1.0 + alpha * (abs_ret_weights - 1.0))
        
        # Apply confidence weights
        if beta > 0:
            confidence = np.maximum(y_pred, 1 - y_pred)
            confidence_weights = 1.0 - beta * (confidence - 0.5) * 2
            confidence_weights = np.clip(confidence_weights, 0.5, 1.5)
            weights *= confidence_weights
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Compute IRM loss
        total_loss, loss_breakdown = irm.irm_loss(y_pred, y_true, environment_masks, weights)
        
        # Add weighting information to breakdown
        loss_breakdown.update({
            'alpha': alpha,
            'beta': beta,
            'mean_weight': weights.mean(),
            'weight_std': weights.std()
        })
        
        return total_loss, loss_breakdown
        
    except Exception as e:
        # Fallback to standard weighted loss
        weighted_loss = enhanced_weighted_logloss(
            y_true, y_pred, returns, sample_weights, alpha, beta, epsilon
        )
        return weighted_loss, {'irm_fallback': True, 'weighted_loss': weighted_loss}

def create_irm_trainer(
    lambda_irm: float = 1.0,
    min_samples_per_env: int = 100,
    invariance_threshold: float = 0.1
):
    """
    Create a trainer function for IRM-enhanced model training.
    
    Args:
        lambda_irm: IRM penalty weight
        min_samples_per_env: Minimum samples per environment
        invariance_threshold: Invariance quality threshold
        
    Returns:
        IRM trainer function
    """
    def irm_trainer(model, X_train, y_train, custom_features, **kwargs):
        """
        IRM-enhanced training function.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            custom_features: Custom regime features
            **kwargs: Additional training parameters
            
        Returns:
            Training results with invariance metrics
        """
        try:
            from .invariant_risk_minimization import FinancialIRM
            
            # Create IRM instance
            irm = FinancialIRM(
                lambda_irm=lambda_irm,
                min_samples_per_env=min_samples_per_env,
                invariance_threshold=invariance_threshold,
                verbose=True
            )
            
            # Create environment masks
            env_masks = irm.create_environment_masks(custom_features)
            
            if len(env_masks) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient environments for IRM training',
                    'environments': list(env_masks.keys())
                }
            
            # For demonstration, we'll use a simple approach
            # In practice, you'd integrate this with your actual model training
            # (LGBM, XGB, CatBoost, etc.)
            
            # Mock training with IRM loss
            # Note: This is a placeholder. In practice, you'd need to
            # integrate IRM loss into your actual training loops
            
            # Generate mock predictions for demonstration
            if hasattr(model, 'predict'):
                predictions = model.predict(X_train)
            else:
                # Mock predictions for demonstration
                predictions = np.random.uniform(0, 1, len(y_train))
            
            # Evaluate invariance
            invariance_metrics = irm.evaluate_invariance(
                predictions, y_train, env_masks
            )
            
            # Check invariance
            is_invariant = irm.is_invariant(
                predictions, y_train, env_masks
            )
            
            results = {
                'success': True,
                'environments': list(env_masks.keys()),
                'lambda_irm': lambda_irm,
                'invariance_metrics': invariance_metrics,
                'is_invariant': is_invariant,
                'model_type': type(model).__name__
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'environments': []
            }
    
    return irm_trainer

def evaluate_model_invariance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    custom_features: pd.DataFrame,
    lambda_irm: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluate model invariance across environments.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        custom_features: Custom regime features
        lambda_irm: IRM penalty weight
        
    Returns:
        Invariance evaluation results
    """
    try:
        from .invariant_risk_minimization import FinancialIRM
        
        # Create IRM instance
        irm = FinancialIRM(lambda_irm=lambda_irm, verbose=True)
        
        # Create environment masks
        env_masks = irm.create_environment_masks(custom_features)
        
        if len(env_masks) < 2:
            return {
                'success': False,
                'error': 'Insufficient environments for invariance evaluation',
                'environments': list(env_masks.keys())
            }
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'predict'):
            predictions = model.predict(X_test)
        else:
            return {
                'success': False,
                'error': 'Model does not have predict method',
                'environments': list(env_masks.keys())
            }
        
        # Evaluate invariance
        invariance_metrics = irm.evaluate_invariance(
            predictions, y_test.values, env_masks
        )
        
        # Check invariance
        is_invariant = irm.is_invariant(
            predictions, y_test.values, env_masks
        )
        
        # Per-environment performance
        env_performance = {}
        for env_name, env_mask in env_masks.items():
            if env_mask.sum() < 100:
                continue
            
            env_pred = predictions[env_mask]
            env_target = y_test.values[env_mask]
            
            try:
                from sklearn.metrics import roc_auc_score, log_loss
                env_auc = roc_auc_score(env_target, env_pred)
                env_loss = log_loss(env_target, env_pred)
                env_performance[env_name] = {
                    'samples': env_mask.sum(),
                    'auc': env_auc,
                    'loss': env_loss
                }
            except Exception:
                env_performance[env_name] = {
                    'samples': env_mask.sum(),
                    'auc': 0.5,
                    'loss': 0.693
                }
        
        results = {
            'success': True,
            'environments': list(env_masks.keys()),
            'lambda_irm': lambda_irm,
            'invariance_metrics': invariance_metrics,
            'is_invariant': is_invariant,
            'env_performance': env_performance,
            'model_type': type(model).__name__
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'environments': []
        }
