#!/usr/bin/env python3
"""
Debug script to isolate the EnhancedPerfectNASRegimeDetector generator error.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import logging

# Add the src directory to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_nas_detector():
    """Test the EnhancedPerfectNASRegimeDetector to isolate the generator error."""
    try:
        logger.info("Starting EnhancedPerfectNASRegimeDetector test...")

        # Import the specific detector class
        from training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
        from training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_config import EnhancedPerfectNASConfig

        # Create a simple configuration
        config = EnhancedPerfectNASConfig(
            n_regimes=3,
            primary_architecture='hybrid',
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_state_space_models=True,
            search_strategy='evolutionary',
            enable_meta_learning=True
        )

        logger.info("Created EnhancedPerfectNASConfig")

        # Create test data (small dataset to avoid memory issues)
        np.random.seed(42)
        n_samples = 100
        n_features = 4  # OHLC data

        # Generate synthetic market data
        test_data = np.random.randn(n_samples, n_features).astype(np.float32)
        timestamps = np.arange(n_samples)

        logger.info(f"Created test data: {test_data.shape}")

        # Create the enhanced detector directly
        logger.info("Creating EnhancedPerfectNASRegimeDetector...")
        detector = EnhancedPerfectNASRegimeDetector(config)
        logger.info("EnhancedPerfectNASRegimeDetector created successfully")

        # Try to detect regimes
        logger.info("Attempting regime detection...")
        result = detector.detect_regimes(test_data, timestamps)
        logger.info(f"Regime detection completed: success={result.success}")

        if result.success:
            logger.info(f"Predictions shape: {result.regime_predictions.shape}")
            logger.info(f"Probabilities shape: {result.regime_probabilities.shape}")
            logger.info(f"Unique regimes: {np.unique(result.regime_predictions)}")
        else:
            logger.error(f"Detection failed: {result.error_message}")

        return result

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    result = test_enhanced_nas_detector()
    if result:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed")
        sys.exit(1)
