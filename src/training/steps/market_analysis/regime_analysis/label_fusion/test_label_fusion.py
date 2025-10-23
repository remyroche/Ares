"""
Test script for the production-ready label fusion implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Import the label fusion modules
from . import (
    RegimeOptimizationService,
    OptimizationMethod,
    FusionMethod,
    OptimizationConfig,
    FusionConfig,
    ValidationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000, n_features: int = 10, n_regimes: int = 3) -> np.ndarray:
    """Create test data for regime optimization."""
    np.random.seed(42)
    
    # Create synthetic data with clear regime structure
    data = np.zeros((n_samples, n_features))
    regime_labels = np.zeros(n_samples, dtype=int)
    
    regime_size = n_samples // n_regimes
    
    for i in range(n_regimes):
        start_idx = i * regime_size
        end_idx = start_idx + regime_size if i < n_regimes - 1 else n_samples
        
        # Create regime-specific data
        regime_center = np.random.randn(n_features) * 2
        regime_data = np.random.randn(end_idx - start_idx, n_features) + regime_center
        
        data[start_idx:end_idx] = regime_data
        regime_labels[start_idx:end_idx] = i
    
    return data, regime_labels


def create_test_labels(n_samples: int = 1000, n_regimes: int = 3, noise_level: float = 0.1) -> List[np.ndarray]:
    """Create test label sets for fusion."""
    np.random.seed(42)
    
    # Create base labels
    base_labels = np.random.randint(0, n_regimes, n_samples)
    
    # Create multiple label sets with different noise levels
    label_sets = []
    
    for i in range(3):
        # Add noise to base labels
        noise_mask = np.random.random(n_samples) < noise_level
        noisy_labels = base_labels.copy()
        noisy_labels[noise_mask] = np.random.randint(0, n_regimes, np.sum(noise_mask))
        
        label_sets.append(noisy_labels)
    
    return label_sets


def test_regime_optimization():
    """Test regime optimization functionality."""
    logger.info("Testing regime optimization...")
    
    # Create test data
    features, true_labels = create_test_data()
    
    # Test different optimization methods
    methods = [
        OptimizationMethod.GRID_SEARCH,
        OptimizationMethod.RANDOM_SEARCH
    ]
    
    for method in methods:
        logger.info(f"Testing {method.value} optimization...")
        
        # Create optimization config
        opt_config = OptimizationConfig(
            method=method,
            n_regimes_range=(2, 5),
            algorithms=['kmeans', 'gmm'],
            max_iterations=20
        )
        
        # Create service
        service = RegimeOptimizationService(optimization_config=opt_config)
        
        # Run optimization
        result = service.optimize_regimes(features)
        
        # Check results
        assert result.success, f"Optimization failed: {result.errors}"
        assert result.n_regimes > 0, "No regimes detected"
        assert result.quality_metrics, "No quality metrics calculated"
        
        logger.info(f"✓ {method.value} optimization successful: {result.n_regimes} regimes, "
                   f"silhouette={result.quality_metrics.get('silhouette_score', 0):.3f}")


def test_label_fusion():
    """Test label fusion functionality."""
    logger.info("Testing label fusion...")
    
    # Create test labels
    label_sets = create_test_labels()
    
    # Test different fusion methods
    methods = [
        FusionMethod.MAJORITY_VOTING,
        FusionMethod.WEIGHTED_AVERAGE,
        FusionMethod.DAWID_SKENE
    ]
    
    for method in methods:
        logger.info(f"Testing {method.value} fusion...")
        
        # Create fusion config
        fusion_config = FusionConfig(method=method)
        
        # Create service
        service = RegimeOptimizationService(fusion_config=fusion_config)
        
        # Run fusion
        result = service.fuse_labels(label_sets)
        
        # Check results
        assert result.success, f"Fusion failed: {result.errors}"
        assert len(result.fused_labels) > 0, "No fused labels produced"
        assert len(result.confidence_scores) > 0, "No confidence scores produced"
        assert result.quality_improvement >= 0, "Negative quality improvement"
        
        logger.info(f"✓ {method.value} fusion successful: "
                   f"quality improvement={result.quality_improvement:.3f}")


def test_regime_validation():
    """Test regime validation functionality."""
    logger.info("Testing regime validation...")
    
    # Create test data
    features, true_labels = create_test_data()
    
    # Create market data for economic validation
    market_data = pd.DataFrame({
        'close': np.random.randn(len(true_labels)).cumsum() + 100,
        'volume': np.random.exponential(1000, len(true_labels)),
        'volatility': np.random.exponential(0.02, len(true_labels))
    })
    
    # Create validation config
    val_config = ValidationConfig(
        min_regime_persistence=0.5,
        min_temporal_stability=0.4,
        min_samples_per_regime=50
    )
    
    # Create service
    service = RegimeOptimizationService(validation_config=val_config)
    
    # Run validation
    result = service.validate_regimes(
        true_labels, 
        features, 
        market_data
    )
    
    # Check results
    assert result.regime_count > 0, "No regimes detected"
    assert result.quality_score >= 0, "Negative quality score"
    assert isinstance(result.regime_statistics, dict), "Invalid regime statistics"
    
    logger.info(f"✓ Validation successful: "
               f"regimes={result.regime_count}, "
               f"quality={result.quality_score:.3f}, "
               f"valid={result.valid}")


def test_integration():
    """Test full integration workflow."""
    logger.info("Testing full integration workflow...")
    
    # Create test data
    features, true_labels = create_test_data()
    label_sets = create_test_labels()
    
    # Create comprehensive config
    opt_config = OptimizationConfig(
        method=OptimizationMethod.GRID_SEARCH,
        n_regimes_range=(2, 4),
        algorithms=['kmeans', 'gmm']
    )
    
    fusion_config = FusionConfig(
        method=FusionMethod.WEIGHTED_AVERAGE
    )
    
    val_config = ValidationConfig(
        min_regime_persistence=0.5,
        min_temporal_stability=0.4
    )
    
    # Create service
    service = RegimeOptimizationService(
        optimization_config=opt_config,
        fusion_config=fusion_config,
        validation_config=val_config
    )
    
    # Step 1: Optimize regimes
    opt_result = service.optimize_regimes(features)
    assert opt_result.success, "Optimization failed"
    
    # Step 2: Fuse labels
    fusion_result = service.fuse_labels(label_sets)
    assert fusion_result.success, "Fusion failed"
    
    # Step 3: Validate fused labels
    val_result = service.validate_regimes(
        fusion_result.fused_labels,
        features
    )
    
    logger.info(f"✓ Integration test successful:")
    logger.info(f"  - Optimized {opt_result.n_regimes} regimes")
    logger.info(f"  - Fused {len(label_sets)} label sets")
    logger.info(f"  - Validation quality: {val_result.quality_score:.3f}")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting label fusion tests...")
    
    try:
        test_regime_optimization()
        test_label_fusion()
        test_regime_validation()
        test_integration()
        
        logger.info("✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)