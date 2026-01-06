"""Confident Learning for Noisy Label Filtering.

This module implements confident learning techniques to identify and handle
noisy labels in the training data, as described in:
"Confident Learning: Estimating Uncertainty in Dataset Labels" (Northcutt et al., 2021)

The core idea is to use out-of-sample predicted probabilities to identify samples
where the model's prediction strongly disagrees with the given label, suggesting
the label may be incorrect.

Key functions:
- compute_label_quality_scores: Compute per-sample quality scores
- identify_label_issues: Find samples with likely label errors
- filter_noisy_labels: Remove or down-weight noisy samples
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, clone
import lightgbm as lgb

from src.utils.tprint import tprint_info, tprint_warning, tprint_success


def compute_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    method: str = "self_confidence",
) -> np.ndarray:
    """
    Compute per-sample label quality scores.
    
    Higher scores indicate higher confidence that the label is correct.
    Lower scores indicate potential label errors.
    
    Args:
        labels: Binary labels (0 or 1), shape (n_samples,)
        pred_probs: Predicted probabilities for class 1, shape (n_samples,)
        method: Scoring method:
            - "self_confidence": P(given_label) - simple and effective
            - "normalized_margin": (P(given_label) - P(other_label)) / 2 + 0.5
            - "entropy_weighted": Self-confidence weighted by prediction entropy
    
    Returns:
        Array of quality scores in [0, 1], shape (n_samples,)
    """
    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)
    pred_probs = np.clip(pred_probs, 1e-6, 1.0 - 1e-6)
    
    n_samples = len(labels)
    if len(pred_probs) != n_samples:
        raise ValueError(f"labels ({n_samples}) and pred_probs ({len(pred_probs)}) must have same length")
    
    # P(given_label): probability assigned to the given label
    p_given = np.where(labels == 1, pred_probs, 1.0 - pred_probs)
    
    if method == "self_confidence":
        # Simple: just use P(given_label)
        scores = p_given
        
    elif method == "normalized_margin":
        # Margin between P(given) and P(other), normalized to [0, 1]
        p_other = 1.0 - p_given
        margin = (p_given - p_other) / 2.0 + 0.5
        scores = margin
        
    elif method == "entropy_weighted":
        # Weight by inverse entropy (high entropy = uncertain = lower quality)
        entropy = -pred_probs * np.log(pred_probs) - (1 - pred_probs) * np.log(1 - pred_probs)
        max_entropy = np.log(2)  # Maximum entropy for binary
        entropy_weight = 1.0 - (entropy / max_entropy)
        scores = p_given * entropy_weight
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.clip(scores, 0.0, 1.0)


def estimate_confident_joint(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate the confident joint matrix Q_y,y* where:
    - y is the given (noisy) label
    - y* is the estimated true label
    
    Args:
        labels: Binary labels (0 or 1)
        pred_probs: Predicted probabilities for class 1
        threshold: Confidence threshold. If None, uses per-class thresholds.
    
    Returns:
        2x2 confident joint matrix
    """
    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)
    
    n_samples = len(labels)
    
    # Compute per-class thresholds if not provided
    if threshold is None:
        # Use average predicted probability per class as threshold
        t0 = np.mean(pred_probs[labels == 0])  # Avg prob for class 0 samples
        t1 = np.mean(pred_probs[labels == 1])  # Avg prob for class 1 samples
        thresholds = [t0, t1]
    else:
        thresholds = [threshold, threshold]
    
    # Initialize confident joint
    C = np.zeros((2, 2), dtype=float)
    
    for given_label in [0, 1]:
        mask = labels == given_label
        probs_subset = pred_probs[mask]
        
        # Predicted label based on threshold
        if given_label == 0:
            # For samples labeled 0, check if model thinks they're 1
            pred_label_1 = probs_subset >= thresholds[0]
            C[0, 0] += np.sum(~pred_label_1)  # Correctly labeled 0
            C[0, 1] += np.sum(pred_label_1)   # Mislabeled (should be 1)
        else:
            # For samples labeled 1, check if model thinks they're 0
            pred_label_0 = probs_subset < thresholds[1]
            C[1, 0] += np.sum(pred_label_0)   # Mislabeled (should be 0)
            C[1, 1] += np.sum(~pred_label_0)  # Correctly labeled 1
    
    # Normalize to get joint distribution
    if C.sum() > 0:
        C = C / C.sum()
    
    return C


def identify_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    method: str = "confident_learning",
    threshold: Optional[float] = None,
    n_to_flag: Optional[int] = None,
    frac_to_flag: Optional[float] = None,
) -> np.ndarray:
    """
    Identify samples with likely label errors.
    
    Args:
        labels: Binary labels (0 or 1)
        pred_probs: Predicted probabilities for class 1
        method: Detection method:
            - "confident_learning": Use confident joint to find off-diagonal samples
            - "low_self_confidence": Flag samples with low P(given_label)
            - "high_loss": Flag samples with high cross-entropy loss
        threshold: Confidence threshold for detection
        n_to_flag: Number of samples to flag (overrides threshold)
        frac_to_flag: Fraction of samples to flag (overrides threshold)
    
    Returns:
        Boolean mask where True indicates a likely label issue
    """
    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)
    pred_probs = np.clip(pred_probs, 1e-6, 1.0 - 1e-6)
    
    n_samples = len(labels)
    
    if method == "confident_learning":
        # Estimate confident joint
        C = estimate_confident_joint(labels, pred_probs, threshold)
        
        # Compute per-class thresholds
        t0 = np.mean(pred_probs[labels == 0]) if (labels == 0).any() else 0.5
        t1 = np.mean(pred_probs[labels == 1]) if (labels == 1).any() else 0.5
        
        # Flag samples in off-diagonal cells
        issues = np.zeros(n_samples, dtype=bool)
        
        # Class 0 samples that model thinks are class 1
        mask_0 = labels == 0
        issues[mask_0] = pred_probs[mask_0] >= t0
        
        # Class 1 samples that model thinks are class 0
        mask_1 = labels == 1
        issues[mask_1] = pred_probs[mask_1] < t1
        
    elif method == "low_self_confidence":
        # Flag samples where P(given_label) is low
        quality_scores = compute_label_quality_scores(labels, pred_probs, "self_confidence")
        
        if threshold is None:
            threshold = 0.5  # Default: flag if model gives <50% to given label
        
        issues = quality_scores < threshold
        
    elif method == "high_loss":
        # Flag samples with high cross-entropy loss
        p_given = np.where(labels == 1, pred_probs, 1.0 - pred_probs)
        loss = -np.log(p_given)
        
        if threshold is None:
            threshold = np.percentile(loss, 90)  # Top 10% highest loss
        
        issues = loss > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Override with n_to_flag or frac_to_flag if specified
    if n_to_flag is not None or frac_to_flag is not None:
        quality_scores = compute_label_quality_scores(labels, pred_probs, "self_confidence")
        
        if frac_to_flag is not None:
            n_to_flag = int(frac_to_flag * n_samples)
        
        n_to_flag = min(n_to_flag, n_samples)
        
        # Flag the n_to_flag samples with lowest quality scores
        threshold_score = np.partition(quality_scores, n_to_flag - 1)[n_to_flag - 1] if n_to_flag > 0 else 0
        issues = quality_scores <= threshold_score
        
        # Ensure exactly n_to_flag samples (handle ties)
        if issues.sum() > n_to_flag:
            issue_indices = np.where(issues)[0]
            scores_at_issues = quality_scores[issue_indices]
            sorted_idx = np.argsort(scores_at_issues)
            keep_idx = issue_indices[sorted_idx[:n_to_flag]]
            issues = np.zeros(n_samples, dtype=bool)
            issues[keep_idx] = True
    
    return issues


def get_cross_val_pred_probs(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    model: Optional[BaseEstimator] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Get out-of-sample predicted probabilities using cross-validation.
    
    Args:
        X: Features
        y: Labels
        model: Classifier with predict_proba method. If None, uses LightGBM.
        n_splits: Number of CV folds
        random_state: Random seed
    
    Returns:
        Out-of-sample predicted probabilities for class 1
    """
    y = np.asarray(y, dtype=int)
    
    if model is None:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            max_depth=6,
            learning_rate=0.1,
            min_data_in_leaf=20,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )
    
    # Handle class imbalance
    n_classes = len(np.unique(y))
    if n_classes < 2:
        tprint_warning("⚠️ Only one class present; returning uniform probabilities")
        return np.full(len(y), 0.5)
    
    # Ensure enough samples per class for stratified CV
    min_class_count = min(np.bincount(y))
    if min_class_count < n_splits:
        n_splits = max(2, min_class_count)
        tprint_warning(f"⚠️ Reduced CV splits to {n_splits} due to class imbalance")
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    try:
        pred_probs = cross_val_predict(
            model, X, y, cv=cv, method='predict_proba', n_jobs=-1
        )[:, 1]
    except Exception as e:
        tprint_warning(f"⚠️ Cross-val prediction failed: {e}. Using simple split.")
        # Fallback: simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=random_state
        )
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        pred_probs = np.full(len(y), 0.5)
        pred_probs[idx_test] = model_clone.predict_proba(X_test)[:, 1]
        # For training samples, use in-sample predictions (less reliable but better than nothing)
        pred_probs[idx_train] = model_clone.predict_proba(X_train)[:, 1]
    
    return pred_probs


def filter_noisy_labels(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    method: str = "confident_learning",
    action: str = "downweight",
    downweight_factor: float = 0.1,
    downweight_scheme: str = "linear_quality",
    frac_to_filter: float = 0.05,
    model: Optional[BaseEstimator] = None,
    n_cv_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Filter or downweight samples with likely label errors.
    
    This is the main entry point for confident learning integration.
    
    Args:
        X: Features
        y: Labels
        sample_weights: Existing sample weights (will be modified)
        method: Detection method (see identify_label_issues)
        action: What to do with flagged samples:
            - "remove": Remove flagged samples entirely
            - "downweight": Reduce weight of flagged samples
            - "flag_only": Just return the flags without modifying
        downweight_factor: Factor to multiply weights by for flagged samples (if action="downweight")
        frac_to_filter: Fraction of samples to flag as noisy
        model: Classifier for probability estimation
        n_cv_splits: Number of CV folds
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of:
        - Modified sample weights (or original if action="flag_only")
        - Boolean mask of flagged samples
        - Diagnostics dict with quality scores and statistics
    """
    y = np.asarray(y, dtype=int)
    n_samples = len(y)
    
    if sample_weights is None:
        sample_weights = np.ones(n_samples, dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float).copy()
    
    if verbose:
        tprint_info(f"🔍 Running confident learning on {n_samples} samples...")
    
    # Step 1: Get out-of-sample predicted probabilities
    pred_probs = get_cross_val_pred_probs(
        X, y, model=model, n_splits=n_cv_splits, random_state=random_state
    )
    
    # Step 2: Compute label quality scores
    quality_scores = compute_label_quality_scores(y, pred_probs, method="self_confidence")
    
    # Step 3: Identify label issues
    n_to_flag = int(frac_to_filter * n_samples)
    issues = identify_label_issues(
        y, pred_probs, method=method, n_to_flag=n_to_flag
    )
    
    n_flagged = int(issues.sum())
    
    # Step 4: Apply action
    if action == "remove":
        # Set weights to 0 for flagged samples
        sample_weights[issues] = 0.0
        if verbose:
            tprint_info(f"🗑️ Removed {n_flagged} samples ({100*n_flagged/n_samples:.1f}%)")
            
    elif action == "downweight":
        # Reduce weights for flagged samples
        if n_flagged > 0:
            try:
                scheme = str(downweight_scheme or "").lower().strip()
            except Exception:
                scheme = "linear_quality"

            if scheme in {"step", "binary"}:
                sample_weights[issues] *= float(downweight_factor)
            else:
                issue_idx = np.where(issues)[0]
                q = np.asarray(quality_scores, dtype=float)
                q_issue = q[issue_idx]
                q_issue = np.where(np.isfinite(q_issue), q_issue, 0.0)

                floor = float(downweight_factor)
                floor = float(np.clip(floor, 0.0, 1.0))

                if scheme in {"linear_rank", "rank"}:
                    # Linear downweight within flagged set based on rank:
                    # - worst-quality flagged sample -> downweight_factor
                    # - best-quality flagged sample  -> 1.0
                    order = np.argsort(q_issue)  # ascending: worst first
                    ranks = np.empty_like(order, dtype=float)
                    if len(order) > 1:
                        ranks[order] = np.linspace(0.0, 1.0, num=len(order), dtype=float)
                    else:
                        ranks[order] = 0.0
                    multipliers = floor + (1.0 - floor) * ranks
                else:
                    # Linear downweight within flagged set based on the quality value:
                    # - min quality in flagged set -> downweight_factor
                    # - max quality in flagged set -> 1.0
                    q_min = float(np.min(q_issue)) if q_issue.size else 0.0
                    q_max = float(np.max(q_issue)) if q_issue.size else 1.0
                    denom = float(q_max - q_min)
                    if denom <= 1e-12:
                        scaled = np.zeros_like(q_issue, dtype=float)
                    else:
                        scaled = (q_issue - q_min) / denom
                    multipliers = floor + (1.0 - floor) * scaled

                multipliers = np.where(np.isfinite(multipliers), multipliers, floor)
                multipliers = np.clip(multipliers, floor, 1.0)
                sample_weights[issue_idx] *= multipliers
        if verbose:
            tprint_info(
                f"⬇️ Downweighted {n_flagged} samples ({100*n_flagged/n_samples:.1f}%) "
                f"with scheme={downweight_scheme} floor={downweight_factor}"
            )
            
    elif action == "flag_only":
        if verbose:
            tprint_info(f"🚩 Flagged {n_flagged} samples ({100*n_flagged/n_samples:.1f}%)")
    else:
        raise ValueError(f"Unknown action: {action}")
    
    # Renormalize weights to maintain mean=1
    if sample_weights.sum() > 0:
        sample_weights *= (n_samples / sample_weights.sum())
    
    # Build diagnostics
    diagnostics = {
        "n_samples": n_samples,
        "n_flagged": n_flagged,
        "frac_flagged": n_flagged / n_samples if n_samples > 0 else 0.0,
        "quality_scores_mean": float(np.mean(quality_scores)),
        "quality_scores_std": float(np.std(quality_scores)),
        "quality_scores_min": float(np.min(quality_scores)),
        "quality_scores_q10": float(np.percentile(quality_scores, 10)),
        "quality_scores_median": float(np.median(quality_scores)),
        "pred_probs_mean": float(np.mean(pred_probs)),
        "pred_probs_std": float(np.std(pred_probs)),
        "confident_joint": estimate_confident_joint(y, pred_probs).tolist(),
        "flagged_class_0": int(issues[y == 0].sum()) if (y == 0).any() else 0,
        "flagged_class_1": int(issues[y == 1].sum()) if (y == 1).any() else 0,
    }
    
    if verbose:
        tprint_success(
            f"✅ Confident learning complete. "
            f"Quality scores: mean={diagnostics['quality_scores_mean']:.3f}, "
            f"min={diagnostics['quality_scores_min']:.3f}"
        )
    
    return sample_weights, issues, diagnostics


def compute_label_noise_estimate(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate the overall label noise rate in the dataset.
    
    Args:
        labels: Binary labels
        pred_probs: Predicted probabilities
    
    Returns:
        Dict with noise estimates per class and overall
    """
    labels = np.asarray(labels, dtype=int)
    pred_probs = np.asarray(pred_probs, dtype=float)
    
    C = estimate_confident_joint(labels, pred_probs)
    
    # Off-diagonal elements represent label errors
    noise_0_to_1 = C[0, 1]  # Labeled 0, should be 1
    noise_1_to_0 = C[1, 0]  # Labeled 1, should be 0
    
    # Class-conditional noise rates
    p_y0 = C[0, :].sum()
    p_y1 = C[1, :].sum()
    
    noise_rate_class_0 = noise_0_to_1 / p_y0 if p_y0 > 0 else 0.0
    noise_rate_class_1 = noise_1_to_0 / p_y1 if p_y1 > 0 else 0.0
    
    overall_noise = noise_0_to_1 + noise_1_to_0
    
    return {
        "overall_noise_rate": float(overall_noise),
        "noise_rate_class_0": float(noise_rate_class_0),
        "noise_rate_class_1": float(noise_rate_class_1),
        "confident_joint": C.tolist(),
    }
