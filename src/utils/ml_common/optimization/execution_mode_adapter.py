"""
Execution Mode Adapter for HPO

This module adjusts HPO parameters based on execution mode (light, blank, full).

Rules:
- light mode: 10% iterations, 2 CV folds
- blank mode: 25% iterations, 5 CV folds (maintained for variance)
- full mode: no change
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Constants for variance validation
MIN_VARIANCE_THRESHOLD = 0.001  # Minimum acceptable variance between CV folds
FOLD_SIMILARITY_THRESHOLD = 0.95  # Maximum similarity allowed between folds


def get_execution_mode() -> str:
    """
    Get the current execution mode from environment or config.
    
    Returns:
        'light', 'blank', 'full', or 'small_dataset'
    """
    # Check environment variable first
    mode = os.environ.get('ARES_EXECUTION_MODE', '').lower()
    
    # Default to 'full' if not set
    if mode not in ['light', 'blank', 'full', 'small_dataset']:
        mode = 'full'
    
    return mode


def adjust_hpo_params_for_mode(
    n_trials: int,
    cv_folds: int,
    execution_mode: Optional[str] = None
) -> tuple[int, int]:
    """
    Adjust HPO parameters based on execution mode.
    
    Args:
        n_trials: Original number of trials
        cv_folds: Original number of CV folds
        execution_mode: Execution mode ('light', 'blank', 'full', 'small_dataset'). If None, auto-detect.
    
    Returns:
        Tuple of (adjusted_n_trials, adjusted_cv_folds)
    """
    if execution_mode is None:
        # Default to the global execution mode
        execution_mode = get_execution_mode()
    else:
        # Reconcile explicit mode with global mode to avoid stale overrides
        global_mode = get_execution_mode()
        # If a caller passes 'light' but the global mode is 'blank' or 'full',
        # trust the global mode so that blank/full runs are not downscaled as light.
        if execution_mode == 'light' and global_mode in ['blank', 'full']:
            logger.info(
                f"ℹ️ Overriding explicit LIGHT execution_mode with global {global_mode.upper()} mode"
            )
            execution_mode = global_mode
    
    if execution_mode == 'light':
        # Light mode: 10% iterations, 2 CV folds
        adjusted_trials = max(1, int(n_trials * 0.1))
        adjusted_folds = 2
        logger.info(f"⚡ LIGHT mode: Reducing HPO trials {n_trials} → {adjusted_trials} (10%)")
        logger.info(f"⚡ LIGHT mode: Reducing CV folds {cv_folds} → {adjusted_folds}")
        logger.warning(f"🔍 LIGHT MODE DIAGNOSTIC: Very small CV folds ({adjusted_folds}) may cause score variance issues")
        
    elif execution_mode == 'blank':
        # Blank mode: 25% iterations, MAINTAIN 5 CV folds (not 3)
        # CRITICAL: Reducing to 3 folds causes near-zero variance and overfitting
        adjusted_trials = max(1, int(n_trials * 0.25))
        adjusted_folds = 5  # Keep 5 folds even in blank mode
        logger.info(f"⚡ BLANK mode: Reducing HPO trials {n_trials} → {adjusted_trials} (25%)")
        logger.info(f"⚡ BLANK mode: MAINTAINING CV folds {cv_folds} → {adjusted_folds} (critical for variance)")
        logger.info(f"🔍 BLANK MODE IMPROVEMENT: Each CV fold will have ~20% of data (maintaining variance)")
        logger.info(f"🔍 VARIANCE PROTECTION: 5-fold CV prevents overfitting and ensures reliable validation")
        
    elif execution_mode == 'small_dataset':
        # Small dataset mode: 5% iterations, 2 CV folds (aggressive optimization for very small datasets)
        adjusted_trials = max(1, int(n_trials * 0.05))
        adjusted_folds = 2
        logger.info(f"⚡ SMALL_DATASET mode: Aggressively reducing HPO trials {n_trials} → {adjusted_trials} (5%)")
        logger.info(f"⚡ SMALL_DATASET mode: Using 2 CV folds for very small datasets")
        
    else:
        # Full mode: no change
        adjusted_trials = n_trials
        adjusted_folds = cv_folds
        logger.info(f"🚀 FULL mode: Using full HPO trials={n_trials}, cv_folds={cv_folds}")
    
    return adjusted_trials, adjusted_folds


def adjust_model_iterations_for_mode(
    iterations: int,
    execution_mode: Optional[str] = None
) -> int:
    """
    Adjust model training iterations (n_estimators, epochs, etc.) based on execution mode.
    
    Args:
        iterations: Original number of iterations
        execution_mode: Execution mode ('light', 'blank', 'full'). If None, auto-detect.
    
    Returns:
        Adjusted number of iterations
    """
    if execution_mode is None:
        execution_mode = get_execution_mode()
    
    if execution_mode == 'light':
        # Light mode: 10% of iterations
        adjusted = max(10, int(iterations * 0.1))
        logger.debug(f"LIGHT mode: Iterations {iterations} → {adjusted} (10%)")
        
    elif execution_mode == 'blank':
        # Blank mode: 25% of iterations
        adjusted = max(25, int(iterations * 0.25))
        logger.debug(f"BLANK mode: Iterations {iterations} → {adjusted} (25%)")
        
    else:
        # Full mode: no change
        adjusted = iterations
    
    return adjusted


def adjust_bootstrap_for_mode(
    n_bootstrap: int,
    execution_mode: Optional[str] = None
) -> int:
    """
    Adjust bootstrap sample count based on execution mode.
    
    Args:
        n_bootstrap: Original number of bootstrap samples
        execution_mode: Execution mode ('light', 'blank', 'full'). If None, auto-detect.
    
    Returns:
        Adjusted number of bootstrap samples
    """
    if execution_mode is None:
        execution_mode = get_execution_mode()
    
    if execution_mode == 'light':
        # Light mode: 10% of bootstrap samples (minimum 5)
        adjusted = max(5, int(n_bootstrap * 0.1))
        logger.info(f"LIGHT mode: Bootstrap samples {n_bootstrap} → {adjusted} (10%)")
        
    elif execution_mode == 'blank':
        # Blank mode: 25% of bootstrap samples (minimum 10)
        adjusted = max(10, int(n_bootstrap * 0.25))
        logger.info(f"BLANK mode: Bootstrap samples {n_bootstrap} → {adjusted} (25%)")
        
    else:
        # Full mode: no change
        adjusted = n_bootstrap
    
    return adjusted


def set_execution_mode(mode: str) -> None:
    """
    Set the execution mode for all subsequent HPO operations.
    
    Args:
        mode: Execution mode ('light', 'blank', 'full')
    """
    if mode not in ['light', 'blank', 'full', 'small_dataset']:
        raise ValueError(f"Invalid execution mode: {mode}. Must be 'light', 'blank', 'full', or 'small_dataset'")
    
    os.environ['ARES_EXECUTION_MODE'] = mode
    logger.info(f"🔧 Execution mode set to: {mode.upper()}")


def validate_cv_variance(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    max_attempts: int = 3
) -> Tuple[bool, float, List[float]]:
    """
    Validate CV fold variance and regenerate if necessary.
    
    Args:
        X: Feature matrix
        y: Target labels
        cv_folds: Number of CV folds to validate
        max_attempts: Maximum attempts to generate valid folds
        
    Returns:
        Tuple of (is_valid, variance_score, fold_scores)
    """
    logger.info(f"🔍 VALIDATING CV VARIANCE with {cv_folds} folds...")
    
    for attempt in range(max_attempts):
        logger.info(f"📊 CV variance validation attempt {attempt + 1}/{max_attempts}")
        
        # Create stratified K-fold splitter
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42 + attempt)
        
        fold_scores = []
        fold_sizes = []
        
        # Calculate scores for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Store fold size information
            fold_sizes.append(len(val_idx))
            
            # Simple baseline model to evaluate fold quality
            # Using majority class as baseline
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            majority_class = unique_classes[np.argmax(class_counts)]
            y_pred = np.full_like(y_val, majority_class)
            
            # Calculate accuracy for this fold
            fold_acc = accuracy_score(y_val, y_pred)
            fold_scores.append(fold_acc)
            
            logger.debug(f"   Fold {fold_idx + 1}: {len(val_idx)} samples, baseline accuracy: {fold_acc:.4f}")
        
        # Calculate variance between folds
        fold_scores = np.array(fold_scores)
        variance_score = float(np.var(fold_scores))
        
        # Log detailed fold information
        logger.info(f"📈 FOLD DISTRIBUTION:")
        logger.info(f"   Fold sizes: {fold_sizes}")
        logger.info(f"   Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        logger.info(f"   Score variance: {variance_score:.8f}")
        logger.info(f"   Score mean: {np.mean(fold_scores):.4f}")
        logger.info(f"   Score std: {np.std(fold_scores):.4f}")
        
        # Check if variance is acceptable
        if variance_score >= MIN_VARIANCE_THRESHOLD:
            logger.info(f"✅ CV VARIANCE VALIDATION PASSED: {variance_score:.8f} >= {MIN_VARIANCE_THRESHOLD}")
            return True, variance_score, fold_scores.tolist()
        else:
            logger.warning(f"❌ CV VARIANCE TOO LOW: {variance_score:.8f} < {MIN_VARIANCE_THRESHOLD}")
            
            # Check for fold similarity
            max_similarity = float(max(fold_scores) - min(fold_scores))
            if max_similarity < (1 - FOLD_SIMILARITY_THRESHOLD):
                logger.warning(f"❌ FOLDS TOO SIMILAR: max difference {max_similarity:.4f} < threshold {(1 - FOLD_SIMILARITY_THRESHOLD):.4f}")
            
            if attempt < max_attempts - 1:
                logger.info(f"🔄 Regenerating folds with different random state...")
    
    # If all attempts failed, return with warning
    logger.error(f"🚨 CV VARIANCE VALIDATION FAILED after {max_attempts} attempts")
    logger.error(f"   Final variance: {variance_score:.8f} (threshold: {MIN_VARIANCE_THRESHOLD})")
    return False, variance_score, fold_scores.tolist()


def log_cv_fold_distribution(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    execution_mode: str = 'full'
) -> None:
    """
    Log detailed information about CV fold distribution.
    
    Args:
        X: Feature matrix
        y: Target labels
        cv_folds: Number of CV folds
        execution_mode: Current execution mode for context
    """
    logger.info(f"📊 CV FOLD DISTRIBUTION ANALYSIS ({execution_mode.upper()} mode)")
    
    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Analyze class distribution in each fold
    unique_classes = np.unique(y)
    class_distributions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_val = y[val_idx]
        
        # Calculate class distribution for this fold
        fold_dist = {}
        for cls in unique_classes:
            count = np.sum(y_val == cls)
            percentage = (count / len(y_val)) * 100
            fold_dist[cls] = {'count': count, 'percentage': percentage}
        
        class_distributions.append(fold_dist)
        
        logger.info(f"   Fold {fold_idx + 1}: {len(val_idx)} samples")
        for cls, stats in fold_dist.items():
            logger.info(f"      Class {cls}: {stats['count']} samples ({stats['percentage']:.1f}%)")
    
    # Check for distribution consistency
    logger.info(f"🔍 DISTRIBUTION CONSISTENCY CHECK:")
    for cls in unique_classes:
        percentages = [fold_dist[cls]['percentage'] for fold_dist in class_distributions]
        mean_pct = np.mean(percentages)
        std_pct = np.std(percentages)
        
        logger.info(f"   Class {cls}: {mean_pct:.1f}% ± {std_pct:.1f}% across folds")
        
        if std_pct > 5.0:  # If variation > 5%
            logger.warning(f"   ⚠️  Class {cls} shows high variation across folds ({std_pct:.1f}%)")
    
    # Overall statistics
    total_samples = len(X)
    samples_per_fold = total_samples / cv_folds
    logger.info(f"📈 OVERALL STATISTICS:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Samples per fold (ideal): {samples_per_fold:.1f}")
    logger.info(f"   Data percentage per fold: {100/cv_folds:.1f}%")
    
    # Variance impact analysis
    if cv_folds < 5:
        logger.warning(f"⚠️  LOW FOLD COUNT IMPACT:")
        logger.warning(f"   Using {cv_folds} folds means each fold has {100/cv_folds:.1f}% of data")
        logger.warning(f"   This reduces variance and may lead to overfitting")
        logger.warning(f"   Recommended minimum: 5 folds (20% data per fold)")
    else:
        logger.info(f"✅ FOLD COUNT OPTIMAL:")
        logger.info(f"   Using {cv_folds} folds provides good variance estimation")
        logger.info(f"   Each fold has {100/cv_folds:.1f}% of data")


def enhanced_adjust_hpo_params_with_validation(
    n_trials: int,
    cv_folds: int,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    execution_mode: Optional[str] = None
) -> Tuple[int, int, bool, float]:
    """
    Enhanced HPO parameter adjustment with CV variance validation.
    
    Args:
        n_trials: Original number of trials
        cv_folds: Original number of CV folds
        X: Feature matrix (optional, for validation)
        y: Target labels (optional, for validation)
        execution_mode: Execution mode ('light', 'blank', 'full'). If None, auto-detect.
    
    Returns:
        Tuple of (adjusted_n_trials, adjusted_cv_folds, is_variance_valid, variance_score)
    """
    if execution_mode is None:
        execution_mode = get_execution_mode()
    
    # Get standard adjusted parameters
    adjusted_trials, adjusted_folds = adjust_hpo_params_for_mode(n_trials, cv_folds, execution_mode)
    
    # If data is provided, validate CV variance
    is_variance_valid = True
    variance_score = 0.0
    
    if X is not None and y is not None:
        logger.info(f"🔍 ENHANCED VALIDATION for {execution_mode.upper()} mode")
        
        # Log fold distribution
        log_cv_fold_distribution(X, y, adjusted_folds, execution_mode)
        
        # Validate variance
        is_variance_valid, variance_score, fold_scores = validate_cv_variance(
            X, y, adjusted_folds
        )
        
        # If variance is too low, try to improve it
        if not is_variance_valid and adjusted_folds < 5:
            logger.warning(f"🔧 ATTEMPTING TO IMPROVE VARIANCE by increasing folds")
            improved_folds = min(5, adjusted_folds + 2)  # Increase by 2, max 5
            
            logger.info(f"📊 Retesting with {improved_folds} folds...")
            is_variance_valid, variance_score, _ = validate_cv_variance(
                X, y, improved_folds
            )
            
            if is_variance_valid:
                adjusted_folds = improved_folds
                logger.info(f"✅ IMPROVED VARIANCE with {adjusted_folds} folds")
            else:
                logger.warning(f"⚠️  Could not achieve sufficient variance even with {improved_folds} folds")
    
    return adjusted_trials, adjusted_folds, is_variance_valid, variance_score

