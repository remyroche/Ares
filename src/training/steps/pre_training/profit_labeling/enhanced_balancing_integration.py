"""
Enhanced Label Balancing & Sample Weighting Integration System

This module provides a comprehensive integration system for the enhanced label balancing
and sample weighting functionality. It serves as a bridge between the core balancing
system and the training pipelines, providing easy-to-use interfaces and automatic
configuration management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import time
from datetime import datetime

# Import the enhanced balancing system
from .label_balancing import (
    ComprehensiveBalancingSystem, LabelBalancer, SampleWeighter,
    BalancingConfig, WeightingConfig, RegimeConfig, ValidationFairnessConfig,
    BalancingTechnique, WeightingScheme,
    DEFAULT_BALANCING_CONFIG, DEFAULT_WEIGHTING_CONFIG, DEFAULT_REGIME_CONFIG, DEFAULT_FAIRNESS_CONFIG
)

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    from src.utils.common_operations import safe_divide, safe_mean, safe_std, validate_dataframe
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False


@dataclass
class BalancingIntegrationConfig:
    """Configuration for balancing system integration."""
    
    # Integration settings
    auto_configure: bool = True
    enable_monitoring: bool = True
    enable_debugging: bool = False
    save_artifacts: bool = True
    
    # Performance settings
    memory_limit_gb: float = 8.0
    max_samples_for_balancing: int = 100000
    enable_parallel_processing: bool = True
    
    # Quality settings
    min_quality_score: float = 0.6
    max_processing_time_seconds: float = 300.0
    
    # Output settings
    output_directory: str = "generated/balancing_artifacts"
    save_reports: bool = True
    save_weights: bool = True


class BalancingIntegrationManager:
    """
    Manager class for integrating enhanced balancing into training pipelines.
    
    This class provides a high-level interface for using the enhanced balancing
    system in training pipelines, with automatic configuration and monitoring.
    """
    
    def __init__(self, config: Optional[BalancingIntegrationConfig] = None):
        """Initialize the balancing integration manager."""
        self.config = config or BalancingIntegrationConfig()
        self.balancing_system = None
        self.monitoring_data = {}
        self.performance_metrics = {}
        
        if TPRINT_AVAILABLE:
            tprint_success("🚀 Enhanced Balancing Integration Manager initialized")
    
    def create_balancing_system(self, 
                               dataset_characteristics: Optional[Dict[str, Any]] = None,
                               custom_config: Optional[Dict[str, Any]] = None) -> ComprehensiveBalancingSystem:
        """
        Create a balancing system with optimal configuration.
        
        Args:
            dataset_characteristics: Optional dataset characteristics for auto-configuration
            custom_config: Optional custom configuration overrides
            
        Returns:
            Configured ComprehensiveBalancingSystem
        """
        if TPRINT_AVAILABLE:
            tprint_info("🔧 Creating enhanced balancing system...")
        
        # Auto-configure based on dataset characteristics
        if self.config.auto_configure and dataset_characteristics:
            balancing_config, weighting_config = self._auto_configure(
                dataset_characteristics
            )
        else:
            balancing_config = DEFAULT_BALANCING_CONFIG
            weighting_config = DEFAULT_WEIGHTING_CONFIG
        
        # Apply custom configuration overrides
        if custom_config:
            balancing_config, weighting_config = self._apply_custom_config(
                balancing_config, weighting_config, custom_config
            )
        
        # Create the comprehensive balancing system
        self.balancing_system = ComprehensiveBalancingSystem(
            balancing_config=balancing_config,
            weighting_config=weighting_config,
            regime_config=DEFAULT_REGIME_CONFIG,
            fairness_config=DEFAULT_FAIRNESS_CONFIG
        )
        
        if TPRINT_AVAILABLE:
            tprint_success("✅ Enhanced balancing system created")
            tprint_info(f"   → Balancing technique: {balancing_config.balancing_technique.value}")
            tprint_info(f"   → Weighting scheme: {weighting_config.weighting_scheme.value}")
        
        return self.balancing_system
    
    def _auto_configure(self, dataset_characteristics: Dict[str, Any]) -> Tuple[BalancingConfig, WeightingConfig]:
        """Auto-configure balancing system based on dataset characteristics."""
        if TPRINT_AVAILABLE:
            tprint_info("🧠 Auto-configuring balancing system...")
        
        # Extract characteristics
        n_samples = dataset_characteristics.get('n_samples', 1000)
        n_classes = dataset_characteristics.get('n_classes', 2)
        imbalance_ratio = dataset_characteristics.get('imbalance_ratio', 0.1)
        has_regime_data = dataset_characteristics.get('has_regime_data', False)
        has_volatility_data = dataset_characteristics.get('has_volatility_data', False)
        dataset_type = dataset_characteristics.get('dataset_type', 'general')
        
        # Configure balancing technique
        if imbalance_ratio < 0.05:
            # Very imbalanced - use SMOTE
            balancing_technique = BalancingTechnique.SMOTE
        elif imbalance_ratio < 0.2:
            # Moderately imbalanced - use hybrid
            balancing_technique = BalancingTechnique.HYBRID
        elif imbalance_ratio < 0.4:
            # Slightly imbalanced - use adaptive
            balancing_technique = BalancingTechnique.ADAPTIVE
        else:
            # Well balanced - use stratified batching
            balancing_technique = BalancingTechnique.STRATIFIED_BATCHING
        
        # Configure weighting scheme
        if has_regime_data and has_volatility_data:
            weighting_scheme = WeightingScheme.INFORMATION_CONTENT
        elif has_regime_data:
            weighting_scheme = WeightingScheme.REGIME_AWARE
        elif has_volatility_data:
            weighting_scheme = WeightingScheme.VOLATILITY
        else:
            weighting_scheme = WeightingScheme.CONFIDENCE
        
        # Create configurations
        balancing_config = BalancingConfig(
            balancing_technique=balancing_technique,
            under_sampling_ratio=min(0.8, max(0.3, 1 - imbalance_ratio)),
            over_sampling_ratio=min(0.5, max(0.1, imbalance_ratio * 2)),
            adaptive_imbalance_threshold=0.1,
            adaptive_min_samples=max(50, n_samples // 100),
            random_state=42
        )
        
        weighting_config = WeightingConfig(
            weighting_scheme=weighting_scheme,
            volatility_window=min(20, max(5, n_samples // 50)),
            confidence_scale=2.0 if dataset_type == 'trading' else 1.5,
            time_decay_half_life=30 if dataset_type == 'trading' else 60,
            regime_frequency_threshold=0.2,
            regime_weight_multiplier=5.0 if has_regime_data else 1.0,
            weight_normalization="l2",
            min_weight=0.1,
            max_weight=10.0
        )
        
        if TPRINT_AVAILABLE:
            tprint_info(f"   → Selected balancing: {balancing_technique.value}")
            tprint_info(f"   → Selected weighting: {weighting_scheme.value}")
        
        return balancing_config, weighting_config
    
    def _apply_custom_config(self, 
                            balancing_config: BalancingConfig,
                            weighting_config: WeightingConfig,
                            custom_config: Dict[str, Any]) -> Tuple[BalancingConfig, WeightingConfig]:
        """Apply custom configuration overrides."""
        # Update balancing config
        for key, value in custom_config.get('balancing', {}).items():
            if hasattr(balancing_config, key):
                setattr(balancing_config, key, value)
        
        # Update weighting config
        for key, value in custom_config.get('weighting', {}).items():
            if hasattr(weighting_config, key):
                setattr(weighting_config, key, value)
        
        return balancing_config, weighting_config
    
    def balance_and_weight_data(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               sample_weight: Optional[pd.Series] = None,
                               additional_features: Optional[Dict[str, pd.Series]] = None,
                               dataset_characteristics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply balancing and weighting to data.
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional existing sample weights
            additional_features: Optional additional features for weighting
            dataset_characteristics: Optional dataset characteristics
            
        Returns:
            Dictionary containing balanced data and metadata
        """
        start_time = time.time()
        
        if TPRINT_AVAILABLE:
            tprint_info("⚖️ Starting enhanced balancing and weighting...")
            tprint_info(f"   → Input samples: {len(X)}")
            tprint_info(f"   → Classes: {y.nunique()}")
            tprint_info(f"   → Imbalance ratio: {y.value_counts().min() / y.value_counts().max():.3f}")
        
        # Create balancing system if not exists
        if self.balancing_system is None:
            self.create_balancing_system(dataset_characteristics)
        
        # Validate input data
        if not self._validate_input_data(X, y):
            raise ValueError("Invalid input data for balancing")
        
        # Check memory constraints
        if len(X) > self.config.max_samples_for_balancing:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Dataset too large ({len(X)} samples), applying sampling...")
            X, y, sample_weight = self._apply_memory_sampling(X, y, sample_weight)
        
        # Apply balancing and weighting
        try:
            X_balanced, y_balanced, final_weights = self.balancing_system.balance_and_weight(
                X, y, sample_weight, additional_features
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = {
                'X_balanced': X_balanced,
                'y_balanced': y_balanced,
                'sample_weights': final_weights,
                'processing_time': processing_time,
                'original_samples': len(X),
                'balanced_samples': len(X_balanced),
                'class_distribution_before': y.value_counts().to_dict(),
                'class_distribution_after': y_balanced.value_counts().to_dict(),
                'weight_statistics': {
                    'mean': float(final_weights.mean()),
                    'std': float(final_weights.std()),
                    'min': float(final_weights.min()),
                    'max': float(final_weights.max()),
                    'median': float(final_weights.median())
                },
                'balancing_technique': self.balancing_system.balancing_config.balancing_technique.value,
                'weighting_scheme': self.balancing_system.weighting_config.weighting_scheme.value,
                'success': True
            }
            
            # Update monitoring data
            self.monitoring_data.update({
                'last_processing_time': processing_time,
                'last_original_samples': len(X),
                'last_balanced_samples': len(X_balanced),
                'last_imbalance_ratio': y.value_counts().min() / y.value_counts().max()
            })
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Balancing completed in {processing_time:.2f}s")
                tprint_info(f"   → Samples: {len(X)} → {len(X_balanced)}")
                tprint_info(f"   → Weight range: [{final_weights.min():.3f}, {final_weights.max():.3f}]")
            
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Balancing failed: {e}")
            
            return {
                'X_balanced': X,
                'y_balanced': y,
                'sample_weights': sample_weight if sample_weight is not None else pd.Series(1.0, index=X.index),
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
    
    def _validate_input_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate input data for balancing."""
        if X.empty or y.empty:
            return False
        
        if len(X) != len(y):
            return False
        
        if y.nunique() < 2:
            return False
        
        return True
    
    def _apply_memory_sampling(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply memory-constrained sampling."""
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        
        X_sampled, _, y_sampled, _, sample_weight_sampled = train_test_split(
            X, y, sample_weight if sample_weight is not None else pd.Series(1.0, index=X.index),
            train_size=self.config.max_samples_for_balancing,
            stratify=y,
            random_state=42
        )
        
        return X_sampled, y_sampled, sample_weight_sampled
    
    def check_validation_fairness(self, 
                                 train_data: Dict[str, Any],
                                 val_data: Dict[str, Any],
                                 live_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check validation fairness."""
        if self.balancing_system is None:
            return {'error': 'Balancing system not initialized'}
        
        return self.balancing_system.check_validation_fairness(train_data, val_data, live_data)
    
    def get_balancing_report(self) -> Dict[str, Any]:
        """Get comprehensive balancing report."""
        return {
            'monitoring_data': self.monitoring_data,
            'performance_metrics': self.performance_metrics,
            'balancing_system_config': {
                'balancing_technique': self.balancing_system.balancing_config.balancing_technique.value if self.balancing_system else None,
                'weighting_scheme': self.balancing_system.weighting_config.weighting_scheme.value if self.balancing_system else None
            },
            'timestamp': datetime.now().isoformat()
        }


# Convenience functions
def create_trading_balancing_manager(config: Optional[BalancingIntegrationConfig] = None) -> BalancingIntegrationManager:
    """Create a trading-optimized balancing manager."""
    return BalancingIntegrationManager(config)


def create_research_balancing_manager(config: Optional[BalancingIntegrationConfig] = None) -> BalancingIntegrationManager:
    """Create a research-optimized balancing manager."""
    return BalancingIntegrationManager(config)


def create_general_balancing_manager(config: Optional[BalancingIntegrationConfig] = None) -> BalancingIntegrationManager:
    """Create a general-purpose balancing manager."""
    return BalancingIntegrationManager(config)


# Example usage and integration patterns
def integrate_with_analyst_training(X: pd.DataFrame, y: pd.Series, 
                                  regime_data: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Example integration with Analyst training pipeline.
    
    Args:
        X: Feature matrix
        y: Target labels (Analyst decisions)
        regime_data: Optional regime assignments
        
    Returns:
        Balanced data ready for Analyst training
    """
    # Create trading-optimized manager
    manager = create_trading_balancing_manager()
    
    # Prepare additional features
    additional_features = {}
    if regime_data is not None:
        additional_features['regime'] = regime_data
    
    # Analyze dataset characteristics
    dataset_characteristics = {
        'n_samples': len(X),
        'n_classes': y.nunique(),
        'imbalance_ratio': y.value_counts().min() / y.value_counts().max(),
        'has_regime_data': regime_data is not None,
        'has_volatility_data': 'volatility' in X.columns or 'returns' in X.columns,
        'dataset_type': 'trading'
    }
    
    # Apply balancing and weighting
    result = manager.balance_and_weight_data(
        X, y, 
        additional_features=additional_features,
        dataset_characteristics=dataset_characteristics
    )
    
    return result


def integrate_with_tactician_training(X: pd.DataFrame, y: pd.Series,
                                    regime_data: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Example integration with Tactician training pipeline.
    
    Args:
        X: Feature matrix
        y: Target labels (Tactician decisions)
        regime_data: Optional regime assignments
        
    Returns:
        Balanced data ready for Tactician training
    """
    # Create trading-optimized manager
    manager = create_trading_balancing_manager()
    
    # Prepare additional features
    additional_features = {}
    if regime_data is not None:
        additional_features['regime'] = regime_data
    
    # Analyze dataset characteristics
    dataset_characteristics = {
        'n_samples': len(X),
        'n_classes': y.nunique(),
        'imbalance_ratio': y.value_counts().min() / y.value_counts().max(),
        'has_regime_data': regime_data is not None,
        'has_volatility_data': 'volatility' in X.columns or 'returns' in X.columns,
        'dataset_type': 'trading'
    }
    
    # Apply balancing and weighting
    result = manager.balance_and_weight_data(
        X, y,
        additional_features=additional_features,
        dataset_characteristics=dataset_characteristics
    )
    
    return result
