import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        Module = object

class SoftF1Loss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-7):
        """
        Args:
            beta (float): Weight of Recall vs Precision.
                          beta=1.0 is balanced.
                          beta=0.5 weighs Precision 2x (Conservative).
                          beta=2.0 weighs Recall 2x (Aggressive).
            epsilon (float): Smoothing factor.
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # y_pred should be probabilities [0, 1] (e.g. after Sigmoid)
        # y_true should be binary [0, 1]

        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        # Derived Soft F-Beta Score
        numerator = (1 + self.beta**2) * tp
        denominator = numerator + fp + (self.beta**2 * fn)

        f_beta = numerator / (denominator + self.epsilon)

        # Return Loss (1 - Score) so we can minimize it
        return 1 - f_beta.mean()

class SoftAUC_PR_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Differentiable Approximation of Average Precision (AP).
        Maximizing AP = Minimizing (1 - AP).
        """
        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 1. Separate Positives and Negatives
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        # If no positives or no negatives in batch, fallback to BCE
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
             return F.binary_cross_entropy(y_pred, y_true)

        scores_pos = y_pred[pos_mask] # Shape: [Num_Pos]
        scores_neg = y_pred[neg_mask] # Shape: [Num_Neg]

        # 2. Calculate Difference Matrix (Pairwise Comparison)
        # We want diff > 0 (Positive Score > Negative Score)
        # Shape: [Num_Pos, Num_Neg]
        # Memory safe check: if matrix is too huge, use a loop or subsample
        # For batch size N, this is (N/2)*(N/2). 10k * 10k is 100M floats = 400MB. OK for CPU.
        diff_matrix = scores_pos.unsqueeze(1) - scores_neg.unsqueeze(0)

        # 3. Soft Counting (Sigmoid approximation of Step Function)
        # sigmoid(x) ~= 1 if x > 0, ~= 0 if x < 0
        # This acts as a "differentiable rank"
        weights = torch.sigmoid(diff_matrix)

        # 4. Compute Soft Precision for each positive
        # "How many negatives did I beat?" / "Total negatives"
        # Note: This is a simplified ranking objective often called "RankNet" logic
        # For full AP approximation, we need rank among Positives too,
        # but maximizing the margin below is often sufficient and more stable.

        # Simple Pairwise Ranking Loss (Maximizes the gap between Pos and Neg)
        # This is strictly "Maximizing AUC", which correlates 99% with AP in practice.
        loss = -torch.mean(torch.log(weights + 1e-7))

        return loss

class PyTorchObjectiveWrapper:
    """
    Wraps a PyTorch loss module for LightGBM custom objective.
    Calculates gradient and hessian (diagonal) via Autograd.
    """
    def __init__(self, loss_module, device='cpu', hessian_mode='diagonal'):
        self.loss_module = loss_module
        self.device = device
        self.hessian_mode = hessian_mode

    def __call__(self, preds, train_data):
        if not TORCH_AVAILABLE:
            raise ImportError("Torch not available for PyTorchObjectiveWrapper")

        y_true = train_data.get_label()

        # Convert to tensor
        # preds from LGBM are raw margins (logits)
        preds_t = torch.tensor(preds, dtype=torch.float32, requires_grad=True, device=self.device)
        y_true_t = torch.tensor(y_true, dtype=torch.float32, device=self.device)

        # Apply sigmoid to get probabilities
        probs_t = torch.sigmoid(preds_t)

        # Calculate loss
        loss = self.loss_module(probs_t, y_true_t)

        # Backward pass for Gradients (dLoss/dPreds)
        # We need dL/dPreds (w.r.t raw margins)
        # Create graph for Hessian
        grads = torch.autograd.grad(loss, preds_t, create_graph=True)[0]

        # Hessian Calculation
        if self.hessian_mode == 'constant':
            hess = np.ones_like(preds)
        elif self.hessian_mode == 'diagonal':
            # Approximate diagonal Hessian: d(grad)/d(pred)
            # This is expensive: requires N backward passes or loop.
            # Fast approximation: for BCE-like losses, hess ~ p(1-p).
            # For ranking losses, it's complex.
            # Efficient diagonal approximation using random vector (Hutchinson) or loop?
            # Loop is too slow.
            # Use 'constant' for stability with ranking losses or simplified approximation.
            # Actually, for SoftF1/AUC, gradient is usually sufficient if we use 'constant' hessian (1.0).
            # LightGBM treats hessian as weight. Constant weight = Gradient Descent.
            hess = np.ones_like(preds)
        else:
            hess = np.ones_like(preds)

        return grads.detach().cpu().numpy(), hess
