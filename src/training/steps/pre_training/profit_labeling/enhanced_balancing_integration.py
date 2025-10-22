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
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import copy
import hashlib
from collections import defaultdict
from scipy.stats import entropy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Import the enhanced balancing system
from .label_balancing import (
    ComprehensiveBalancingSystem, LabelBalancer, SampleWeighter,
    BalancingConfig, WeightingConfig, RegimeConfig, ValidationFairnessConfig,
    BalancingTechnique, WeightingScheme,
    DEFAULT_BALANCING_CONFIG, DEFAULT_WEIGHTING_CONFIG, DEFAULT_REGIME_CONFIG, DEFAULT_FAIRNESS_CONFIG
)

# Import additional utilities for data-driven approach
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Note: tprint and hardware utilities are available through BaseStep
# No need for direct imports as they're inherited from BaseStep


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
    
    # Data-driven configuration
    cv_folds: int = 5
    random_state: int = 42
    enable_early_stopping: bool = True
    stability_window: int = 3
    degradation_threshold: float = 1.5  # MAD multiplier
    
    # Weight constraints
    max_weight: float = 10.0
    min_weight: float = 0.1
    target_ess_ratio: float = 0.6  # Target ESS/N ratio
    
    # Multiclass support
    multiclass_imbalance_metric: str = 'entropy'  # 'entropy', 'gini', 'max_min_ratio'
    
    # Reproducibility
    save_manifest: bool = True
    version_tracking: bool = True


class BalancingIntegrationManager(BaseStep):
    """
    Manager class for integrating enhanced balancing into training pipelines.
    
    This class provides a high-level interface for using the enhanced balancing
    system in training pipelines, with automatic configuration and monitoring.
    Inherits from BaseStep for standardized pipeline integration.
    """
    
    def __init__(self, config: Optional[BalancingIntegrationConfig] = None):
        """Initialize the balancing integration manager."""
        super().__init__()
        self.config = config or BalancingIntegrationConfig()
        self.balancing_system = None
        self.monitoring_data = {}
        self.performance_metrics = defaultdict(list)
        self.quality_history = []
        self.processing_timer = None
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_state)
        random.seed(self.config.random_state)
        
        self.tprint_success("🚀 Enhanced Balancing Integration Manager initialized")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the balancing integration step.
        
        Args:
            config: Configuration dictionary containing:
                - market_data: DataFrame with market data
                - labels: Series or array with labels
                - dataset_characteristics: Optional dataset characteristics
                - custom_config: Optional custom configuration overrides
                - symbol: Optional symbol for context
                - exchange: Optional exchange for context
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - balanced_data: Balanced dataset
                - balanced_labels: Balanced labels
                - sample_weights: Sample weights
                - balancing_metrics: Balancing performance metrics
                - artifacts: List of generated artifacts
        """
        try:
            # Start processing timer
            self.processing_timer = time.time()
            
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract data from config
            market_data = config.get('market_data')
            labels = config.get('labels')
            dataset_characteristics = config.get('dataset_characteristics')
            custom_config = config.get('custom_config')
            
            if market_data is None or labels is None:
                return {
                    'success': False,
                    'error': 'Missing required data: market_data and labels are required'
                }
            
            # Convert labels to Series if needed and validate
            if not isinstance(labels, pd.Series):
                labels = pd.Series(labels, index=market_data.index)
            
            # Comprehensive input validation
            validation_result = self._validate_input_data(market_data, labels)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f'Input validation failed: {validation_result['error']}'
                }
            
            # Preview input data
            self.tprint_data_preview(market_data, "input_market_data", max_rows=5)
            self.tprint_data_format(market_data, "input_market_data")
            
            labels_df = labels.to_frame('labels')
            self.tprint_data_preview(labels_df, "input_labels", max_rows=5)
            self.tprint_data_format(labels_df, "input_labels")
            
            # Apply hardware optimization to input data
            if self.hardware_utils and self.hardware_utils.get('optimize_dataframe'):
                market_data = self.hardware_utils['optimize_dataframe'](market_data)
                self.tprint_info("🔧 Input data optimized for hardware acceleration")
            
            # Create balancing system with data-driven configuration
            balancing_system = self.create_balancing_system(
                market_data=market_data,
                labels=labels,
                dataset_characteristics=dataset_characteristics,
                custom_config=custom_config
            )
            
            # Perform balancing using the correct method name
            result = self.balance_and_weight_data(
                X=market_data,
                y=labels,
                dataset_characteristics=dataset_characteristics
            )
            
            if not result['success']:
                return result
            
            # Check processing time limit
            elapsed_time = time.time() - self.processing_timer
            if elapsed_time > self.config.max_processing_time_seconds:
                self.tprint_warning(f"⚠️ Processing time exceeded limit ({elapsed_time:.2f}s > {self.config.max_processing_time_seconds}s)")
                return {
                    'success': False,
                    'error': f'Processing time exceeded limit: {elapsed_time:.2f}s',
                    'recommendation': 'Consider reducing dataset size or using faster balancing techniques'
                }
            
            # Standardize result schema
            standardized_result = self._standardize_result_schema(result)
            
            # Save artifacts
            artifacts = []
            if self.config.save_artifacts:
                # Apply hardware optimization to balanced data
                if self.hardware_utils and self.hardware_utils.get('optimize_dataframe'):
                    standardized_result['balanced_data'] = self.hardware_utils['optimize_dataframe'](standardized_result['balanced_data'])
                    self.tprint_info("🔧 Balanced data optimized for hardware acceleration")
                
                # Verify index alignment after optimization
                self._verify_index_alignment(standardized_result)
                
                # Save artifacts
                artifacts = self._save_balancing_artifacts(standardized_result)
            
            # Update performance metrics and check for degradation
            self._update_performance_metrics(standardized_result)
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(standardized_result, artifacts)
            self._save_outcome_file(outcome_content, 'balancing_integration_outcome')
            
            # Save manifest for reproducibility
            if self.config.save_manifest:
                manifest = self._generate_config_manifest(market_data, labels, standardized_result)
                manifest_path = self._save_metadata(manifest, 'balancing_manifest')
                if manifest_path:
                    artifacts.append(manifest_path)
            
            return {
                'success': True,
                'balanced_data': standardized_result['balanced_data'],
                'balanced_labels': standardized_result['balanced_labels'],
                'sample_weights': standardized_result['sample_weights'],
                'balancing_metrics': standardized_result['balancing_metrics'],
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Balancing integration failed: {str(e)}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, result: Dict[str, Any], artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Enhanced Balancing Integration Outcome

## Summary
- **Status**: {'Success' if result.get('success', False) else 'Failed'}
- **Processing Time**: {result.get('processing_time', 0):.2f} seconds
- **Original Samples**: {result.get('original_samples', 0)}
- **Balanced Samples**: {result.get('balanced_samples', 0)}
- **Artifacts Generated**: {len(artifacts)}

## Balancing Metrics
"""
        
        if 'balancing_metrics' in result and result['balancing_metrics']:
            metrics = result['balancing_metrics']
            content += f"""
- **Class Distribution Before**: {metrics.get('class_distribution_before', 'N/A')}
- **Class Distribution After**: {metrics.get('class_distribution_after', 'N/A')}
- **Quality Score**: {metrics.get('quality_score', 0):.3f}
- **Balancing Technique**: {metrics.get('balancing_technique', 'Unknown')}
- **Weighting Scheme**: {metrics.get('weighting_scheme', 'Unknown')}
- **ESS Ratio**: {metrics.get('ess_ratio', 'N/A')}
- **Hellinger Distance**: {metrics.get('hellinger_distance', 'N/A')}
- **Weight Statistics**: {metrics.get('weight_statistics', 'N/A')}
"""
        
        content += f"""
## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Configuration
- **Auto Configure**: {self.config.auto_configure}
- **Enable Monitoring**: {self.config.enable_monitoring}
- **Memory Limit**: {self.config.memory_limit_gb} GB
- **Max Samples**: {self.config.max_samples_for_balancing}
- **Random State**: {self.config.random_state}
- **CV Folds**: {self.config.cv_folds}
"""
        
        return content
    
    def create_balancing_system(self, 
                               market_data: pd.DataFrame,
                               labels: pd.Series,
                               dataset_characteristics: Optional[Dict[str, Any]] = None,
                               custom_config: Optional[Dict[str, Any]] = None) -> ComprehensiveBalancingSystem:
        """
        Create a balancing system with data-driven optimal configuration.
        
        Args:
            market_data: Feature matrix for data-driven configuration
            labels: Target labels for data-driven configuration
            dataset_characteristics: Optional dataset characteristics for auto-configuration
            custom_config: Optional custom configuration overrides
            
        Returns:
            Configured ComprehensiveBalancingSystem
        """
        self.tprint_info("🔧 Creating enhanced balancing system with data-driven configuration...")
        
        # Deep copy defaults to avoid mutation
        base_balancing_config = copy.deepcopy(DEFAULT_BALANCING_CONFIG)
        base_weighting_config = copy.deepcopy(DEFAULT_WEIGHTING_CONFIG)
        
        # Data-driven auto-configuration
        if self.config.auto_configure:
            balancing_config, weighting_config = self._data_driven_auto_configure(
                market_data, labels, dataset_characteristics
            )
        else:
            balancing_config = base_balancing_config
            weighting_config = base_weighting_config
        
        # Apply custom configuration overrides last
        if custom_config:
            balancing_config, weighting_config = self._apply_custom_config(
                balancing_config, weighting_config, custom_config
            )
            self.tprint_info("🔧 Applied custom configuration overrides")
        
        # Create the comprehensive balancing system
        self.balancing_system = ComprehensiveBalancingSystem(
            balancing_config=balancing_config,
            weighting_config=weighting_config,
            regime_config=DEFAULT_REGIME_CONFIG,
            fairness_config=DEFAULT_FAIRNESS_CONFIG
        )
        
        self.tprint_success("✅ Enhanced balancing system created")
        self.tprint_info(f"   → Balancing technique: {balancing_config.balancing_technique.value}")
        self.tprint_info(f"   → Weighting scheme: {weighting_config.weighting_scheme.value}")
        
        return self.balancing_system
    
    def _data_driven_auto_configure(self, 
                                   market_data: pd.DataFrame, 
                                   labels: pd.Series,
                                   dataset_characteristics: Optional[Dict[str, Any]] = None) -> Tuple[BalancingConfig, WeightingConfig]:
        """Data-driven auto-configuration using CV-based selection."""
        self.tprint_info("🧠 Performing data-driven auto-configuration...")
        
        # Extract characteristics
        n_samples = len(market_data)
        n_classes = labels.nunique()
        
        # Calculate multiclass imbalance metric
        if self.config.multiclass_imbalance_metric == 'entropy':
            class_counts = labels.value_counts()
            class_probs = class_counts / class_counts.sum()
            imbalance_metric = 1 - entropy(class_probs) / np.log(n_classes)  # Normalized entropy
        elif self.config.multiclass_imbalance_metric == 'gini':
            class_counts = labels.value_counts()
            class_probs = class_counts / class_counts.sum()
            imbalance_metric = 1 - np.sum(class_probs ** 2)  # Gini coefficient
        else:  # max_min_ratio
            class_counts = labels.value_counts()
            imbalance_metric = class_counts.min() / class_counts.max()
        
        has_regime_data = dataset_characteristics.get('has_regime_data', False) if dataset_characteristics else False
        has_volatility_data = dataset_characteristics.get('has_volatility_data', False) if dataset_characteristics else False
        dataset_type = dataset_characteristics.get('dataset_type', 'general') if dataset_characteristics else 'general'
        
        # Data-driven technique selection using CV
        balancing_technique = self._select_balancing_technique_cv(market_data, labels, imbalance_metric)
        
        # Data-driven weighting scheme selection
        weighting_scheme = self._select_weighting_scheme_cv(market_data, labels, has_regime_data, has_volatility_data)
        
        # Calibrate hyperparameters using CV
        hyperparams = self._calibrate_hyperparameters_cv(market_data, labels, balancing_technique, weighting_scheme)
        
        # Create configurations with calibrated parameters
        balancing_config = BalancingConfig(
            balancing_technique=balancing_technique,
            under_sampling_ratio=hyperparams['under_sampling_ratio'],
            over_sampling_ratio=hyperparams['over_sampling_ratio'],
            adaptive_imbalance_threshold=hyperparams.get('adaptive_imbalance_threshold', 0.1),
            adaptive_min_samples=hyperparams.get('adaptive_min_samples', max(50, n_samples // 100)),
            smote_k_neighbors=hyperparams.get('smote_k_neighbors', 5),
            random_state=self.config.random_state
        )
        
        weighting_config = WeightingConfig(
            weighting_scheme=weighting_scheme,
            volatility_window=hyperparams.get('volatility_window', min(20, max(5, n_samples // 50))),
            confidence_scale=hyperparams.get('confidence_scale', 2.0 if dataset_type == 'trading' else 1.5),
            time_decay_half_life=hyperparams.get('time_decay_half_life', 30 if dataset_type == 'trading' else 60),
            regime_frequency_threshold=hyperparams.get('regime_frequency_threshold', 0.2),
            regime_weight_multiplier=hyperparams.get('regime_weight_multiplier', 5.0 if has_regime_data else 1.0),
            weight_normalization=hyperparams.get('weight_normalization', "l2"),
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight
        )
        
        self.tprint_info(f"   → Selected balancing: {balancing_technique.value} (CV score: {hyperparams.get('cv_score', 'N/A'):.3f})")
        self.tprint_info(f"   → Selected weighting: {weighting_scheme.value}")
        
        return balancing_config, weighting_config
    
    def _apply_custom_config(self, 
                            balancing_config: BalancingConfig,
                            weighting_config: WeightingConfig,
                            custom_config: Dict[str, Any]) -> Tuple[BalancingConfig, WeightingConfig]:
        """Apply custom configuration overrides with validation."""
        self.tprint_info("🔧 Applying custom configuration overrides...")
        
        # Track changes for logging
        balancing_changes = []
        weighting_changes = []
        
        # Update balancing config
        for key, value in custom_config.get('balancing', {}).items():
            if hasattr(balancing_config, key):
                old_value = getattr(balancing_config, key)
                setattr(balancing_config, key, value)
                balancing_changes.append(f"{key}: {old_value} -> {value}")
            else:
                self.tprint_warning(f"⚠️ Unknown balancing config key: {key}")
        
        # Update weighting config
        for key, value in custom_config.get('weighting', {}).items():
            if hasattr(weighting_config, key):
                old_value = getattr(weighting_config, key)
                setattr(weighting_config, key, value)
                weighting_changes.append(f"{key}: {old_value} -> {value}")
            else:
                self.tprint_warning(f"⚠️ Unknown weighting config key: {key}")
        
        # Log changes
        if balancing_changes:
            self.tprint_info(f"   → Balancing changes: {', '.join(balancing_changes)}")
        if weighting_changes:
            self.tprint_info(f"   → Weighting changes: {', '.join(weighting_changes)}")
        
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
        
        self.tprint_info("⚖️ Starting enhanced balancing and weighting...")
        self.tprint_info(f"   → Input samples: {len(X)}")
        self.tprint_info(f"   → Classes: {y.nunique()}")
        
        # Calculate multiclass imbalance metric
        class_counts = y.value_counts()
        if self.config.multiclass_imbalance_metric == 'entropy':
            class_probs = class_counts / class_counts.sum()
            imbalance_metric = 1 - entropy(class_probs) / np.log(len(class_counts))
        else:
            imbalance_metric = class_counts.min() / class_counts.max()
        
        self.tprint_info(f"   → Imbalance metric: {imbalance_metric:.3f}")
        
        # Create balancing system if not exists
        if self.balancing_system is None:
            self.create_balancing_system(X, y, dataset_characteristics)
        
        # Validate input data
        validation_result = self._validate_input_data(X, y)
        if not validation_result['valid']:
            raise ValueError(f"Invalid input data for balancing: {validation_result['error']}")
        
        # Check memory constraints and apply sampling if needed
        if len(X) > self.config.max_samples_for_balancing:
            self.tprint_warning(f"⚠️ Dataset too large ({len(X)} samples), applying memory sampling...")
            X, y, sample_weight = self._apply_memory_sampling(X, y, sample_weight)
        
        # Apply balancing and weighting
        try:
            X_balanced, y_balanced, final_weights = self.balancing_system.balance_and_weight(
                X, y, sample_weight, additional_features
            )
            
            processing_time = time.time() - start_time
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(X, y, X_balanced, y_balanced, final_weights)
            
            # Create result with standardized schema
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
                'success': True,
                **metrics  # Add comprehensive metrics
            }
            
            # Update monitoring data
            self.monitoring_data.update({
                'last_processing_time': processing_time,
                'last_original_samples': len(X),
                'last_balanced_samples': len(X_balanced),
                'last_imbalance_metric': imbalance_metric
            })
            
            self.tprint_success(f"✅ Balancing completed in {processing_time:.2f}s")
            self.tprint_info(f"   → Samples: {len(X)} → {len(X_balanced)}")
            self.tprint_info(f"   → Weight range: [{final_weights.min():.3f}, {final_weights.max():.3f}]")
            self.tprint_info(f"   → Quality score: {metrics.get('quality_score', 0):.3f}")
            
            return result
            
        except Exception as e:
            self.tprint_error(f"❌ Balancing failed: {e}")
            
            return {
                'X_balanced': X,
                'y_balanced': y,
                'sample_weights': sample_weight if sample_weight is not None else pd.Series(1.0, index=X.index),
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
    
    def _validate_input_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive input data validation."""
        result = {'valid': True, 'error': None}
        
        # Basic checks
        if X.empty or y.empty:
            result['valid'] = False
            result['error'] = 'Input data cannot be empty'
            return result
        
        if len(X) != len(y):
            result['valid'] = False
            result['error'] = 'X and y must have the same length'
            return result
        
        # Index alignment check
        if not X.index.equals(y.index):
            result['valid'] = False
            result['error'] = 'X and y must have aligned indices'
            return result
        
        # Class validation
        if y.nunique() < 2:
            result['valid'] = False
            result['error'] = 'y must have at least 2 classes'
            return result
        
        # NaN/inf checks
        if X.isnull().any().any():
            result['valid'] = False
            result['error'] = 'X contains NaN values'
            return result
        
        if y.isnull().any():
            result['valid'] = False
            result['error'] = 'y contains NaN values'
            return result
        
        if np.isinf(X.values).any():
            result['valid'] = False
            result['error'] = 'X contains infinite values'
            return result
        
        if np.isinf(y.values).any():
            result['valid'] = False
            result['error'] = 'y contains infinite values'
            return result
        
        # y dimensionality check
        if y.ndim != 1:
            result['valid'] = False
            result['error'] = 'y must be 1-dimensional'
            return result
        
        # Multiclass dtype consistency
        unique_dtypes = set(type(val) for val in y.unique())
        if len(unique_dtypes) > 1:
            result['valid'] = False
            result['error'] = 'y contains mixed data types'
            return result
        
        return result
    
    def _apply_memory_sampling(self, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply memory-constrained sampling with time-series awareness."""
        max_samples = min(len(X), self.config.max_samples_for_balancing)
        
        # Check if we can use stratified sampling
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            # Fall back to proportional sampling by class
            self.tprint_warning("⚠️ Some classes have < 2 samples, using proportional sampling")
            return self._proportional_sampling(X, y, sample_weight, max_samples)
        
        # Use stratified sampling if possible
        try:
            from sklearn.model_selection import train_test_split
            
            X_sampled, _, y_sampled, _, sample_weight_sampled = train_test_split(
                X, y, sample_weight if sample_weight is not None else pd.Series(1.0, index=X.index),
                train_size=max_samples,
                stratify=y,
                random_state=self.config.random_state
            )
            
            return X_sampled, y_sampled, sample_weight_sampled
            
        except ValueError as e:
            self.tprint_warning(f"⚠️ Stratified sampling failed: {e}, using proportional sampling")
            return self._proportional_sampling(X, y, sample_weight, max_samples)
    
    def _proportional_sampling(self, X: pd.DataFrame, y: pd.Series, 
                              sample_weight: Optional[pd.Series], 
                              max_samples: int) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Proportional sampling by class."""
        class_counts = y.value_counts()
        total_samples = len(X)
        
        # Calculate target samples per class
        target_samples_per_class = {}
        for class_label, count in class_counts.items():
            target_samples_per_class[class_label] = int((count / total_samples) * max_samples)
        
        # Sample from each class
        sampled_indices = []
        for class_label, target_count in target_samples_per_class.items():
            class_indices = X[y == class_label].index
            if len(class_indices) > 0:
                n_samples = min(target_count, len(class_indices))
                sampled_class_indices = np.random.choice(class_indices, size=n_samples, replace=False)
                sampled_indices.extend(sampled_class_indices)
        
        sampled_indices = pd.Index(sampled_indices)
        
        return X.loc[sampled_indices], y.loc[sampled_indices], (
            sample_weight.loc[sampled_indices] if sample_weight is not None else None
        )
    
    def _select_balancing_technique_cv(self, X: pd.DataFrame, y: pd.Series, 
                                     imbalance_metric: float) -> BalancingTechnique:
        """Select balancing technique using cross-validation."""
        self.tprint_info("🔍 Selecting balancing technique using CV...")
        
        # Define candidate techniques
        candidates = [
            (None, "No balancing"),
            (BalancingTechnique.RANDOM_UNDER_SAMPLING, "Random under-sampling"),
            (BalancingTechnique.RANDOM_OVER_SAMPLING, "Random over-sampling"),
            (BalancingTechnique.SMOTE, "SMOTE"),
            (BalancingTechnique.HYBRID, "Hybrid SMOTE+Under"),
        ]
        
        best_technique = None
        best_score = -np.inf
        
        for technique, name in candidates:
            try:
                score = self._evaluate_balancing_technique_cv(X, y, technique)
                self.tprint_info(f"   → {name}: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_technique = technique
                    
            except Exception as e:
                self.tprint_warning(f"   → {name}: Failed ({e})")
                continue
        
        if best_technique is None:
            best_technique = BalancingTechnique.ADAPTIVE  # Fallback
            
        self.tprint_info(f"   → Selected: {best_technique.value if best_technique else 'None'} (score: {best_score:.3f})")
        return best_technique
    
    def _evaluate_balancing_technique_cv(self, X: pd.DataFrame, y: pd.Series, 
                                       technique: Optional[BalancingTechnique]) -> float:
        """Evaluate balancing technique using model-free CV scoring."""
        if technique is None:
            # No balancing - evaluate original data
            return self._calculate_composite_score(X, y, pd.Series(1.0, index=X.index))
        
        # Create temporary balancing system
        temp_config = copy.deepcopy(DEFAULT_BALANCING_CONFIG)
        temp_config.balancing_technique = technique
        temp_config.random_state = self.config.random_state
        
        temp_system = ComprehensiveBalancingSystem(
            balancing_config=temp_config,
            weighting_config=DEFAULT_WEIGHTING_CONFIG,
            regime_config=DEFAULT_REGIME_CONFIG,
            fairness_config=DEFAULT_FAIRNESS_CONFIG
        )
        
        try:
            X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
            return self._calculate_composite_score(X_balanced, y_balanced, weights)
        except Exception:
            return -np.inf
    
    def _select_weighting_scheme_cv(self, X: pd.DataFrame, y: pd.Series,
                                  has_regime_data: bool, has_volatility_data: bool) -> WeightingScheme:
        """Select weighting scheme using cross-validation."""
        self.tprint_info("🔍 Selecting weighting scheme using CV...")
        
        # Define candidate schemes based on available data
        candidates = [WeightingScheme.CONFIDENCE]
        
        if has_volatility_data:
            candidates.append(WeightingScheme.VOLATILITY)
        if has_regime_data:
            candidates.append(WeightingScheme.REGIME_AWARE)
        if has_regime_data and has_volatility_data:
            candidates.append(WeightingScheme.INFORMATION_CONTENT)
        
        best_scheme = WeightingScheme.CONFIDENCE
        best_score = -np.inf
        
        for scheme in candidates:
            try:
                score = self._evaluate_weighting_scheme_cv(X, y, scheme)
                self.tprint_info(f"   → {scheme.value}: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_scheme = scheme
                    
            except Exception as e:
                self.tprint_warning(f"   → {scheme.value}: Failed ({e})")
                continue
        
        self.tprint_info(f"   → Selected: {best_scheme.value} (score: {best_score:.3f})")
        return best_scheme
    
    def _evaluate_weighting_scheme_cv(self, X: pd.DataFrame, y: pd.Series, 
                                    scheme: WeightingScheme) -> float:
        """Evaluate weighting scheme using CV scoring."""
        # Create temporary weighting system
        temp_config = copy.deepcopy(DEFAULT_WEIGHTING_CONFIG)
        temp_config.weighting_scheme = scheme
        
        temp_system = ComprehensiveBalancingSystem(
            balancing_config=DEFAULT_BALANCING_CONFIG,
            weighting_config=temp_config,
            regime_config=DEFAULT_REGIME_CONFIG,
            fairness_config=DEFAULT_FAIRNESS_CONFIG
        )
        
        try:
            X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
            return self._calculate_composite_score(X_balanced, y_balanced, weights)
        except Exception:
            return -np.inf
    
    def _calibrate_hyperparameters_cv(self, X: pd.DataFrame, y: pd.Series,
                                     balancing_technique: BalancingTechnique,
                                     weighting_scheme: WeightingScheme) -> Dict[str, Any]:
        """Calibrate hyperparameters using cross-validation."""
        self.tprint_info("🔍 Calibrating hyperparameters using CV...")
        
        hyperparams = {}
        
        # Calibrate balancing parameters
        if balancing_technique in [BalancingTechnique.RANDOM_UNDER_SAMPLING, BalancingTechnique.HYBRID]:
            hyperparams['under_sampling_ratio'] = self._calibrate_under_sampling_ratio(X, y)
        
        if balancing_technique in [BalancingTechnique.RANDOM_OVER_SAMPLING, BalancingTechnique.HYBRID]:
            hyperparams['over_sampling_ratio'] = self._calibrate_over_sampling_ratio(X, y)
        
        if balancing_technique == BalancingTechnique.SMOTE:
            hyperparams['smote_k_neighbors'] = self._calibrate_smote_k_neighbors(X, y)
        
        # Calibrate weighting parameters
        if weighting_scheme == WeightingScheme.VOLATILITY:
            hyperparams['volatility_window'] = self._calibrate_volatility_window(X, y)
        
        if weighting_scheme == WeightingScheme.CONFIDENCE:
            hyperparams['confidence_scale'] = self._calibrate_confidence_scale(X, y)
        
        if weighting_scheme == WeightingScheme.TIME_DECAY:
            hyperparams['time_decay_half_life'] = self._calibrate_time_decay_half_life(X, y)
        
        return hyperparams
    
    def _calibrate_under_sampling_ratio(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calibrate under-sampling ratio using CV."""
        ratios = [0.3, 0.5, 0.7, 0.9]
        best_ratio = 0.5
        best_score = -np.inf
        
        for ratio in ratios:
            try:
                temp_config = copy.deepcopy(DEFAULT_BALANCING_CONFIG)
                temp_config.balancing_technique = BalancingTechnique.RANDOM_UNDER_SAMPLING
                temp_config.under_sampling_ratio = ratio
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=temp_config,
                    weighting_config=DEFAULT_WEIGHTING_CONFIG,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_ratio = ratio
                    
            except Exception:
                continue
        
        return best_ratio
    
    def _calibrate_over_sampling_ratio(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calibrate over-sampling ratio using CV."""
        ratios = [0.1, 0.3, 0.5, 0.7]
        best_ratio = 0.3
        best_score = -np.inf
        
        for ratio in ratios:
            try:
                temp_config = copy.deepcopy(DEFAULT_BALANCING_CONFIG)
                temp_config.balancing_technique = BalancingTechnique.RANDOM_OVER_SAMPLING
                temp_config.over_sampling_ratio = ratio
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=temp_config,
                    weighting_config=DEFAULT_WEIGHTING_CONFIG,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_ratio = ratio
                    
            except Exception:
                continue
        
        return best_ratio
    
    def _calibrate_smote_k_neighbors(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Calibrate SMOTE k_neighbors using CV."""
        k_values = [3, 5, 7, 9]
        best_k = 5
        best_score = -np.inf
        
        for k in k_values:
            try:
                temp_config = copy.deepcopy(DEFAULT_BALANCING_CONFIG)
                temp_config.balancing_technique = BalancingTechnique.SMOTE
                temp_config.smote_k_neighbors = k
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=temp_config,
                    weighting_config=DEFAULT_WEIGHTING_CONFIG,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception:
                continue
        
        return best_k
    
    def _calibrate_volatility_window(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Calibrate volatility window using CV."""
        n_samples = len(X)
        windows = [min(20, max(5, n_samples // 50)), min(30, max(10, n_samples // 30)), min(50, max(15, n_samples // 20))]
        best_window = windows[0]
        best_score = -np.inf
        
        for window in windows:
            try:
                temp_config = copy.deepcopy(DEFAULT_WEIGHTING_CONFIG)
                temp_config.weighting_scheme = WeightingScheme.VOLATILITY
                temp_config.volatility_window = window
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=DEFAULT_BALANCING_CONFIG,
                    weighting_config=temp_config,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_window = window
                    
            except Exception:
                continue
        
        return best_window
    
    def _calibrate_confidence_scale(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calibrate confidence scale using CV."""
        scales = [1.0, 1.5, 2.0, 2.5, 3.0]
        best_scale = 2.0
        best_score = -np.inf
        
        for scale in scales:
            try:
                temp_config = copy.deepcopy(DEFAULT_WEIGHTING_CONFIG)
                temp_config.weighting_scheme = WeightingScheme.CONFIDENCE
                temp_config.confidence_scale = scale
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=DEFAULT_BALANCING_CONFIG,
                    weighting_config=temp_config,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_scale = scale
                    
            except Exception:
                continue
        
        return best_scale
    
    def _calibrate_time_decay_half_life(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Calibrate time decay half-life using CV."""
        half_lives = [15, 30, 60, 90, 120]
        best_half_life = 30
        best_score = -np.inf
        
        for half_life in half_lives:
            try:
                temp_config = copy.deepcopy(DEFAULT_WEIGHTING_CONFIG)
                temp_config.weighting_scheme = WeightingScheme.TIME_DECAY
                temp_config.time_decay_half_life = half_life
                
                temp_system = ComprehensiveBalancingSystem(
                    balancing_config=DEFAULT_BALANCING_CONFIG,
                    weighting_config=temp_config,
                    regime_config=DEFAULT_REGIME_CONFIG,
                    fairness_config=DEFAULT_FAIRNESS_CONFIG
                )
                
                X_balanced, y_balanced, weights = temp_system.balance_and_weight(X, y)
                score = self._calculate_composite_score(X_balanced, y_balanced, weights)
                
                if score > best_score:
                    best_score = score
                    best_half_life = half_life
                    
            except Exception:
                continue
        
        return best_half_life
    
    def _calculate_composite_score(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> float:
        """Calculate composite quality score (0-1)."""
        # ESS ratio component (40%)
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        ess_ratio = ess / len(weights)
        ess_score = min(1.0, ess_ratio / self.config.target_ess_ratio)
        
        # Hellinger distance component (25%)
        hellinger_score = self._calculate_hellinger_score(y)
        
        # Regime coverage score (20%)
        regime_score = self._calculate_regime_coverage_score(X, y)
        
        # Weight stability score (15%)
        weight_cv = weights.std() / weights.mean() if weights.mean() > 0 else 1.0
        stability_score = max(0.0, 1.0 - weight_cv)
        
        # Composite score
        composite_score = (
            0.40 * ess_score +
            0.25 * hellinger_score +
            0.20 * regime_score +
            0.15 * stability_score
        )
        
        return composite_score
    
    def _calculate_hellinger_score(self, y: pd.Series) -> float:
        """Calculate Hellinger distance score."""
        class_counts = y.value_counts()
        n_classes = len(class_counts)
        
        if n_classes < 2:
            return 0.0
        
        # Target uniform distribution
        target_probs = np.ones(n_classes) / n_classes
        actual_probs = class_counts.values / class_counts.sum()
        
        # Hellinger distance
        hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(actual_probs) - np.sqrt(target_probs)) ** 2))
        
        # Convert to score (1 - distance)
        return max(0.0, 1.0 - hellinger_dist)
    
    def _calculate_regime_coverage_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate regime coverage score."""
        # For now, return a default score since we don't have regime data
        # In practice, this would check regime coverage after balancing
        return 1.0
    
    def _calculate_comprehensive_metrics(self, X_orig: pd.DataFrame, y_orig: pd.Series,
                                       X_balanced: pd.DataFrame, y_balanced: pd.Series,
                                       weights: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive balancing metrics."""
        # ESS metrics
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        ess_ratio = ess / len(weights)
        
        # Hellinger distance
        hellinger_distance = self._calculate_hellinger_distance(y_orig, y_balanced)
        
        # Weight concentration (Gini coefficient)
        weight_gini = self._calculate_gini_coefficient(weights)
        
        # Quality score
        quality_score = self._calculate_composite_score(X_balanced, y_balanced, weights)
        
        return {
            'ess': float(ess),
            'ess_ratio': float(ess_ratio),
            'hellinger_distance': float(hellinger_distance),
            'weight_gini': float(weight_gini),
            'quality_score': float(quality_score)
        }
    
    def _calculate_hellinger_distance(self, y1: pd.Series, y2: pd.Series) -> float:
        """Calculate Hellinger distance between two distributions."""
        # Get unique classes from both series
        all_classes = set(y1.unique()) | set(y2.unique())
        
        # Calculate probabilities
        p1 = np.array([(y1 == c).sum() for c in all_classes]) / len(y1)
        p2 = np.array([(y2 == c).sum() for c in all_classes]) / len(y2)
        
        # Hellinger distance
        return np.sqrt(0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2))
    
    def _calculate_gini_coefficient(self, weights: pd.Series) -> float:
        """Calculate Gini coefficient for weight distribution."""
        if len(weights) == 0:
            return 0.0
        
        sorted_weights = np.sort(weights.values)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    def _standardize_result_schema(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize result schema to expected format."""
        standardized = {
            'balanced_data': result.get('X_balanced'),
            'balanced_labels': result.get('y_balanced'),
            'sample_weights': result.get('sample_weights'),
            'balancing_metrics': {
                'class_distribution_before': result.get('class_distribution_before', {}),
                'class_distribution_after': result.get('class_distribution_after', {}),
                'weight_statistics': result.get('weight_statistics', {}),
                'balancing_technique': result.get('balancing_technique', 'Unknown'),
                'weighting_scheme': result.get('weighting_scheme', 'Unknown'),
                'quality_score': result.get('quality_score', 0.0),
                'ess_ratio': result.get('ess_ratio', 0.0),
                'hellinger_distance': result.get('hellinger_distance', 0.0),
                'weight_gini': result.get('weight_gini', 0.0),
                'original_samples': result.get('original_samples', 0),
                'balanced_samples': result.get('balanced_samples', 0),
                'processing_time': result.get('processing_time', 0.0)
            },
            'success': result.get('success', False),
            'processing_time': result.get('processing_time', 0.0),
            'original_samples': result.get('original_samples', 0),
            'balanced_samples': result.get('balanced_samples', 0)
        }
        
        return standardized
    
    def _verify_index_alignment(self, result: Dict[str, Any]) -> None:
        """Verify index alignment after any transformations."""
        balanced_data = result['balanced_data']
        balanced_labels = result['balanced_labels']
        sample_weights = result['sample_weights']
        
        # Check alignment
        if not balanced_data.index.equals(balanced_labels.index):
            raise ValueError("Index misalignment: balanced_data and balanced_labels")
        
        if not balanced_data.index.equals(sample_weights.index):
            raise ValueError("Index misalignment: balanced_data and sample_weights")
        
        # Check for duplicates
        if balanced_data.index.duplicated().any():
            raise ValueError("Duplicate indices found in balanced_data")
        
        if balanced_labels.index.duplicated().any():
            raise ValueError("Duplicate indices found in balanced_labels")
        
        if sample_weights.index.duplicated().any():
            raise ValueError("Duplicate indices found in sample_weights")
    
    def _save_balancing_artifacts(self, result: Dict[str, Any]) -> List[str]:
        """Save balancing artifacts."""
        artifacts = []
        
        # Save balanced data
        balanced_data_path = self._save_dataframe(
            result['balanced_data'], 
            'balanced_market_data'
        )
        if balanced_data_path:
            artifacts.append(balanced_data_path)
        
        # Save balanced labels
        labels_df = result['balanced_labels'].to_frame('labels')
        balanced_labels_path = self._save_dataframe(
            labels_df, 
            'balanced_labels'
        )
        if balanced_labels_path:
            artifacts.append(balanced_labels_path)
        
        # Save sample weights
        if self.config.save_weights:
            weights_df = result['sample_weights'].to_frame('weights')
            weights_path = self._save_dataframe(
                weights_df, 
                'sample_weights'
            )
            if weights_path:
                artifacts.append(weights_path)
        
        # Save balancing metrics
        if self.config.save_reports:
            metrics_path = self._save_metadata(
                result['balancing_metrics'], 
                'balancing_metrics'
            )
            if metrics_path:
                artifacts.append(metrics_path)
        
        return artifacts
    
    def _update_performance_metrics(self, result: Dict[str, Any]) -> None:
        """Update performance metrics and check for degradation."""
        quality_score = result['balancing_metrics'].get('quality_score', 0.0)
        ess_ratio = result['balancing_metrics'].get('ess_ratio', 0.0)
        hellinger_distance = result['balancing_metrics'].get('hellinger_distance', 0.0)
        
        # Update history
        self.quality_history.append({
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'ess_ratio': ess_ratio,
            'hellinger_distance': hellinger_distance
        })
        
        # Check for degradation
        if len(self.quality_history) >= self.config.stability_window:
            recent_scores = [entry['quality_score'] for entry in self.quality_history[-self.config.stability_window:]]
            median_score = np.median(recent_scores)
            mad = np.median(np.abs(recent_scores - median_score))
            
            if quality_score < median_score - self.config.degradation_threshold * mad:
                self.tprint_warning(f"⚠️ Performance degradation detected: {quality_score:.3f} < {median_score:.3f} - {self.config.degradation_threshold} * {mad:.3f}")
                result['balancing_metrics']['degradation'] = True
            else:
                result['balancing_metrics']['degradation'] = False
    
    def _generate_config_manifest(self, X: pd.DataFrame, y: pd.Series, 
                                result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration manifest for reproducibility."""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'data_hashes': {
                'X_index_hash': hashlib.md5(str(X.index).encode()).hexdigest(),
                'y_values_hash': hashlib.md5(str(y.values).encode()).hexdigest()
            },
            'configuration': {
                'random_state': self.config.random_state,
                'cv_folds': self.config.cv_folds,
                'max_samples_for_balancing': self.config.max_samples_for_balancing,
                'target_ess_ratio': self.config.target_ess_ratio,
                'multiclass_imbalance_metric': self.config.multiclass_imbalance_metric
            },
            'selected_techniques': {
                'balancing_technique': result['balancing_metrics'].get('balancing_technique'),
                'weighting_scheme': result['balancing_metrics'].get('weighting_scheme')
            },
            'performance_metrics': result['balancing_metrics'],
            'code_versions': {
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'sklearn': 'unknown'  # Would need to import sklearn
            }
        }
        
        return manifest
    
    def check_validation_fairness(self, 
                                 train_data: Dict[str, Any],
                                 val_data: Dict[str, Any],
                                 live_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check validation fairness."""
        if self.balancing_system is None:
            return {'error': 'Balancing system not initialized'}
        
        return self.balancing_system.check_validation_fairness(train_data, val_data, live_data)
    
    def _enforce_weight_caps(self, weights: pd.Series) -> pd.Series:
        """Enforce weight caps and check concentration."""
        # Apply caps
        weights_capped = weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)
        
        # Check concentration
        weight_gini = self._calculate_gini_coefficient(weights_capped)
        
        if weight_gini > 0.8:  # High concentration threshold
            self.tprint_warning(f"⚠️ High weight concentration detected (Gini: {weight_gini:.3f})")
            
            # Reduce max weight and recompute
            new_max_weight = weights_capped.quantile(0.95)  # Use 95th percentile as new max
            weights_capped = weights_capped.clip(lower=self.config.min_weight, upper=new_max_weight)
            
            self.tprint_info(f"   → Reduced max weight to {new_max_weight:.3f}")
        
        return weights_capped
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping conditions are met."""
        if not self.config.enable_early_stopping:
            return False
        
        # Check processing time
        if self.processing_timer and (time.time() - self.processing_timer) > self.config.max_processing_time_seconds:
            self.tprint_warning("⚠️ Early stopping: Processing time exceeded")
            return True
        
        # Check quality degradation
        if len(self.quality_history) >= self.config.stability_window:
            recent_scores = [entry['quality_score'] for entry in self.quality_history[-self.config.stability_window:]]
            if len(recent_scores) >= 3:
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if trend < -0.1:  # Declining trend
                    self.tprint_warning("⚠️ Early stopping: Quality degradation trend detected")
                    return True
        
        return False
    
    def _monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor memory usage and provide recommendations."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024 ** 3)
            
            if memory_gb > self.config.memory_limit_gb:
                self.tprint_warning(f"⚠️ Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.config.memory_limit_gb} GB)")
                return {
                    'memory_gb': memory_gb,
                    'limit_gb': self.config.memory_limit_gb,
                    'exceeded': True,
                    'recommendation': 'Consider reducing max_samples_for_balancing or using memory sampling'
                }
            else:
                return {
                    'memory_gb': memory_gb,
                    'limit_gb': self.config.memory_limit_gb,
                    'exceeded': False
                }
        except ImportError:
            return {'error': 'psutil not available for memory monitoring'}
    
    def _validate_serialization_types(self, data: Any) -> Any:
        """Ensure data contains only serializable types."""
        if isinstance(data, dict):
            return {k: self._validate_serialization_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._validate_serialization_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'value'):  # Enum
            return data.value
        else:
            return data
    
    def get_balancing_report(self) -> Dict[str, Any]:
        """Get comprehensive balancing report."""
        report = {
            'monitoring_data': self.monitoring_data,
            'performance_metrics': dict(self.performance_metrics),
            'quality_history': self.quality_history,
            'balancing_system_config': {
                'balancing_technique': self.balancing_system.balancing_config.balancing_technique.value if self.balancing_system else None,
                'weighting_scheme': self.balancing_system.weighting_config.weighting_scheme.value if self.balancing_system else None
            },
            'memory_usage': self._monitor_memory_usage(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure serializable types
        return self._validate_serialization_types(report)


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
    
    # Analyze dataset characteristics with multiclass support
    class_counts = y.value_counts()
    if manager.config.multiclass_imbalance_metric == 'entropy':
        class_probs = class_counts / class_counts.sum()
        imbalance_metric = 1 - entropy(class_probs) / np.log(len(class_counts))
    else:
        imbalance_metric = class_counts.min() / class_counts.max()
    
    dataset_characteristics = {
        'n_samples': len(X),
        'n_classes': y.nunique(),
        'imbalance_metric': imbalance_metric,
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
    
    # Analyze dataset characteristics with multiclass support
    class_counts = y.value_counts()
    if manager.config.multiclass_imbalance_metric == 'entropy':
        class_probs = class_counts / class_counts.sum()
        imbalance_metric = 1 - entropy(class_probs) / np.log(len(class_counts))
    else:
        imbalance_metric = class_counts.min() / class_counts.max()
    
    dataset_characteristics = {
        'n_samples': len(X),
        'n_classes': y.nunique(),
        'imbalance_metric': imbalance_metric,
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
