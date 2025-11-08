"""
Execution Mode Adapter for HPO

This module adjusts HPO parameters based on execution mode (light, blank, full).

Rules:
- light mode: 10% iterations, 2 CV folds
- blank mode: 25% iterations, 3 CV folds  
- full mode: no change
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
        execution_mode = get_execution_mode()
    
    if execution_mode == 'light':
        # Light mode: 10% iterations, 2 CV folds
        adjusted_trials = max(1, int(n_trials * 0.1))
        adjusted_folds = 2
        logger.info(f"⚡ LIGHT mode: Reducing HPO trials {n_trials} → {adjusted_trials} (10%)")
        logger.info(f"⚡ LIGHT mode: Reducing CV folds {cv_folds} → {adjusted_folds}")
        
    elif execution_mode == 'blank':
        # Blank mode: 25% iterations, 3 CV folds
        adjusted_trials = max(1, int(n_trials * 0.25))
        adjusted_folds = 3
        logger.info(f"⚡ BLANK mode: Reducing HPO trials {n_trials} → {adjusted_trials} (25%)")
        logger.info(f"⚡ BLANK mode: Reducing CV folds {cv_folds} → {adjusted_folds}")
        
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

