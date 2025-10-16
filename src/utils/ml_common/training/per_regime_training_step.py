"""
Per-Regime Training Step - Enhanced with Overfitting Prevention and Lookahead Bias Detection

Base class for per-regime training steps with comprehensive ML utilities integration.
Enhanced Features:
- Purged cross-validation for temporal data integrity
- Early stopping for all supported models
- Lookahead bias detection and prevention
- Enhanced regularization parameters
- Overfitting monitoring and detection
- Walk-forward validation
- Ensemble diversity metrics
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import logging
import time
from datetime import datetime, timedelta
import warnings

from .base_training_step import BaseTrainingStep
from ..config.base_training_config import PerRegimeTrainingConfig

# Enhanced training utilities are now available from BaseTrainingStep
# No need to import separately - use self.enhanced_training_available

# Universal validation integration
try:
    from ..universal_validation_integration import (
        get_validation_integrator,
        ValidationIntegrationConfig,
        intelligently_select_utilities,
        start_monitoring_session,
        monitor_training_step,
        perform_data_leakage_check,
        perform_enhanced_validation,
        perform_complexity_analysis
    )
    UNIVERSAL_VALIDATION_AVAILABLE = True
except ImportError:
    UNIVERSAL_VALIDATION_AVAILABLE = False
    get_validation_integrator = None
    ValidationIntegrationConfig = None
    intelligently_select_utilities = None
    start_monitoring_session = None
    monitor_training_step = None
    perform_data_leakage_check = None
    perform_enhanced_validation = None
    perform_complexity_analysis = None

logger = logging.getLogger(__name__)

class PerRegimeTrainingStep(BaseTrainingStep):
    """
    Enhanced base class for per-regime training steps with comprehensive ML utilities.

    This class provides common functionality for training models on a per-regime basis,
    including regime analysis, data preparation, and per-regime model training.

    Enhanced Features:
    - Overfitting prevention and detection
    - Lookahead bias detection and prevention
    - Enhanced regularization parameters
    - Early stopping for all supported models
    - Purged cross-validation for temporal data
    - Walk-forward validation
    - Ensemble diversity monitoring
    """

    def __init__(self, config: PerRegimeTrainingConfig):
        """
        Initialize enhanced per-regime training step.

        Args:
            config: Per-regime training configuration
        """
        super().__init__(config)
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)

        # Per-regime specific results
        self.regime_models = {}
        self.regime_metadata = {}

        # Enhanced training utilities
        self.enhanced_training_available = ENHANCED_TRAINING_AVAILABLE
        self.training_enhancer = None
        self.enhanced_training_config = None

        # Universal validation integration
        self.universal_validation_available = UNIVERSAL_VALIDATION_AVAILABLE
        self.validation_integrator = None
        self.validation_config = None

        # Initialize enhanced training utilities (inherited from BaseTrainingStep)
        # Enhanced training utilities are already available from base class
        # Can access via self.enhanced_training_utils, self.training_enhancer, etc.

        # Initialize universal validation integration
        if self.universal_validation_available:
            self._initialize_universal_validation_integration()

        self.logger.info("✅ Enhanced Per-Regime Training Step initialized")

    def _initialize_enhanced_training_utilities(self):
        """Initialize enhanced training utilities for per-regime training (inherited from BaseTrainingStep)."""
        # Enhanced training utilities are already available from base class
        # Can access via self.enhanced_training_utils, self.training_enhancer, etc.
        self.logger.info("✅ Enhanced training utilities initialized for per-regime training")
        try:
            # Create enhanced training configuration
            self.enhanced_training_config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                model_type='auto'
            )

            # Initialize training enhancer
            self.training_enhancer = TrainingStepEnhancer(self.enhanced_training_config)

            self.logger.info("✅ Enhanced training utilities initialized successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced training utilities initialization failed: {e}")
            self.enhanced_training_config = None
            self.training_enhancer = None

    def _initialize_universal_validation_integration(self):
        """Initialize universal validation integration for comprehensive model validation."""
        try:
            # Create validation integration configuration
            self.validation_config = ValidationIntegrationConfig(
                enable_validation=True,
                enable_overfitting_detection=True,
                enable_temporal_validation=True,
                enable_timeframe_validation=True,
                enable_data_leakage_prevention=True,
                enable_overfitting_monitoring=True,
                enable_enhanced_validation=True,
                enable_hpo_overfitting_prevention=True,
                enable_model_complexity_analysis=True,
                save_validation_reports=True,
                validation_report_directory="reports/validation/per_regime",
                enable_validation_logging=True,
                fail_on_validation_error=False,
                warn_on_validation_issues=True,
                auto_select_utilities=True
            )

            # Initialize validation integrator
            self.validation_integrator = get_validation_integrator(self.validation_config)

            self.logger.info("✅ Universal validation integration initialized successfully")

        except Exception as e:
            self.logger.warning(f"⚠️ Universal validation integration initialization failed: {e}")
            self.validation_config = None
            self.validation_integrator = None

    def train_regime_models(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train models for each regime with enhanced overfitting prevention.

        Args:
            regime_data: Prepared data for each regime
            feature_names: Names of input features
            timestamps: Optional timestamps for temporal validation

        Returns:
            Dictionary containing training results for each regime
        """
        # Use enhanced training if available
        if self.enhanced_training_available and self.training_enhancer:
            self.logger.info("🚀 Using enhanced training with overfitting prevention")
            return self._train_regime_models_enhanced(regime_data, feature_names, timestamps)
        else:
            self.logger.info("🔄 Using standard training (enhanced utilities not available)")
            return self._train_regime_models_standard(regime_data, feature_names)

    def _train_regime_models_enhanced(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Train models with enhanced overfitting prevention and lookahead bias detection."""
        results = {
            'models': {},
            'metadata': {},
            'regime_analysis': {},
            'enhanced_training_metadata': {},
            'overfitting_warnings': [],
            'ensemble_diversity': None
        }

        trained_models = []

        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue

            self.logger.info(f"🎯 Training enhanced models for regime {regime} ({data['samples']} samples)...")

            # Extract data for this regime
            X = data['X']
            y = data['y']
            timestamps_regime = timestamps.get(regime) if timestamps else None

            # Validate temporal data for lookahead bias
            if timestamps_regime is not None:
                self.logger.info(f"🔍 Validating temporal data for regime {regime}...")
                is_valid, warnings = self.training_enhancer.enhanced_utils.validate_temporal_data(
                    X, y, timestamps_regime, strict_mode=True
                )
                if warnings:
                    for warning in warnings:
                        self.logger.warning(f"⚠️ {warning}")
                        results['overfitting_warnings'].append(f"Regime {regime}: {warning}")
                if not is_valid:
                    self.logger.error(f"❌ Temporal data validation failed for regime {regime}")
                    continue

            # Train each model type for this regime
            regime_model_results = {}

            for model_type in self.config.model_types:
                try:
                    self.logger.info(f"🔄 Training {model_type} for regime {regime} with enhanced utilities...")

                    # Create model instance
                    model = self.training_utils.create_model(model_type)

                    # Apply enhanced regularization
                    model = self.training_enhancer.enhanced_utils.apply_enhanced_regularization(
                        model, model_type
                    )

                    # Train with early stopping and overfitting monitoring
                    trained_model, metadata = self.training_enhancer.enhance_training_step(
                        X, y, model, timestamps_regime, f"{model_type}_regime_{regime}"
                    )

                    regime_model_results[model_type] = {
                        'model': trained_model,
                        'metadata': metadata
                    }
                    trained_models.append(trained_model)

                    # Check for overfitting warnings
                    if metadata.get('overfitting_detected', False):
                        warning_msg = f"Overfitting detected in {model_type} for regime {regime}"
                        results['overfitting_warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to train {model_type} for regime {regime}: {e}")
                    continue

            results['models'][regime] = regime_model_results

            # Add regime analysis
            results['regime_analysis'][regime] = {
                'samples': data['samples'],
                'features': X.shape[1] if len(X.shape) > 1 else 0,
                'models_trained': len(regime_model_results),
                'overfitting_detected': any(
                    results['models'][regime][mt]['metadata'].get('overfitting_detected', False)
                    for mt in results['models'][regime]
                )
            }

        # Calculate ensemble diversity if multiple models
        if len(trained_models) > 1:
            self.logger.info("📊 Calculating ensemble diversity...")
            # Combine all regime data for diversity calculation
            all_X = np.vstack([data['X'] for data in regime_data.values() if not data.get('use_global', False)])
            all_y = np.hstack([data['y'] for data in regime_data.values() if not data.get('use_global', False)])

            diversity_metrics = self.training_enhancer.enhanced_utils.calculate_ensemble_diversity(
                trained_models, all_X, all_y
            )
            results['ensemble_diversity'] = diversity_metrics

            if diversity_metrics.get('diversity_score', 0) < 0.1:
                self.logger.warning("⚠️ Low ensemble diversity detected")
            else:
                self.logger.info("✅ Good ensemble diversity")

        # Add enhanced training metadata
        results['enhanced_training_metadata'] = {
            'overfitting_prevention_enabled': True,
            'lookahead_bias_detection_enabled': timestamps is not None,
            'early_stopping_enabled': True,
            'enhanced_regularization_enabled': True,
            'temporal_validation_enabled': timestamps is not None,
            'total_warnings': len(results['overfitting_warnings']),
            'regimes_processed': len([r for r in regime_data.values() if not r.get('use_global', False)])
        }

        self.logger.info("✅ Enhanced regime training completed successfully")
        return results

    def _train_regime_models_standard(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train models using standard training (fallback)."""
        regime_models = {}
        regime_metadata = {}

        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue

            self.logger.info(f"🔄 Training models for regime {regime} ({data['samples']} samples)...")

            # Train each model type for this regime
            regime_model_results = {}

            for model_type in self.config.model_types:
                self.logger.info(f"🔄 Training {model_type} for regime {regime}...")

                # Perform HPO if enabled
                if self.config.enable_hpo:
                    search_space = self.config.hpo_search_spaces.get(model_type, {})
                    optimized_model = self.training_utils.optimize_model_with_hpo(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        search_space=search_space,
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )
                else:
                    optimized_model = self.training_utils.train_single_model(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )

                regime_model_results[model_type] = optimized_model

            regime_models[regime] = regime_model_results

            # Store regime metadata
            regime_metadata[regime] = {
                'samples': data['samples'],
                'augmented': data['augmented'],
                'hmm_states': data.get('hmm_states'),
                'models_trained': list(regime_model_results.keys()),
                'training_time': time.time()
            }

            self.logger.info(f"✅ Regime {regime} models trained: {list(regime_model_results.keys())}")

        return {
            'models': regime_models,
            'metadata': regime_metadata
        }

    def evaluate_regime_models(
        self,
        regime_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance per regime.

        Args:
            regime_results: Training results for each regime
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results per regime
        """
        return self.evaluation_utils.evaluate_regime_performance(
            models={regime: {model_type: result['model'] for model_type, result in models.items()}
                   for regime, models in regime_results['models'].items()},
            X=X,
            y=y,
            regime_labels=regime_labels,
            metrics=self.config.evaluation_metrics,
            is_classification=is_classification
        )

    def save_regime_models(
        self,
        regime_results: Dict[str, Any],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[int, List[str]]:
        """
        Save trained models for each regime.

        Args:
            regime_results: Training results for each regime
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier

        Returns:
            Dictionary containing saved model paths for each regime
        """
        saved_paths = {}

        for regime, models in regime_results['models'].items():
            # Extract models from results
            model_dict = {model_type: result['model'] for model_type, result in models.items()}

            # Save models for this regime
            model_paths = self.save_models(
                models=model_dict,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )

            saved_paths[regime] = model_paths

            # Save regime metadata
            regime_metadata = regime_results['metadata'][regime]
            self.save_metadata(
                metadata=regime_metadata,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )

        return saved_paths

    def train_regime_models_vectorized(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Train models for all regimes simultaneously using batch processing.

        This method optimizes the entire per-regime training pipeline by:
        - Pre-computing shared computations across regimes
        - Batch processing multiple model types simultaneously
        - Parallel HPO across regimes and model types
        - Memory-efficient processing with shared data structures

        Args:
            regime_data: Prepared data for each regime
            feature_names: Names of input features

        Returns:
            Dictionary containing training results for all regimes
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_time = time.time()
        self.logger.info("🚀 VECTORIZED: Starting batch regime model training...")

        regime_models = {}
        regime_metadata = {}
        processed_regimes = 0

        # VECTORIZED: Group regimes by data size for optimal processing
        large_regimes = {}
        small_regimes = {}

        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue

            if data['samples'] > 10000:
                large_regimes[regime] = data
            else:
                small_regimes[regime] = data

        # VECTORIZED: Process large regimes with parallel model training
        if large_regimes:
            self.logger.info(f"🧠 VECTORIZED: Processing {len(large_regimes)} large regimes with parallel training")
            large_results = self._train_large_regimes_parallel(large_regimes, feature_names)
            regime_models.update(large_results['models'])
            regime_metadata.update(large_results['metadata'])
            processed_regimes += len(large_regimes)

        # VECTORIZED: Process small regimes with batch processing
        if small_regimes:
            self.logger.info(f"📊 VECTORIZED: Processing {len(small_regimes)} small regimes with batch training")
            small_results = self._train_small_regimes_batch(small_regimes, feature_names)
            regime_models.update(small_results['models'])
            regime_metadata.update(small_results['metadata'])
            processed_regimes += len(small_regimes)

        processing_time = time.time() - start_time
        self.logger.info(f"✅ Per-regime training completed in {processing_time:.2f}s")
        return {
            'models': regime_models,
            'metadata': regime_metadata,
            'processing_time': processing_time,
            'vectorized': True
        }

    def _train_large_regimes_parallel(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Train large regimes with parallel processing."""

        results = {'models': {}, 'metadata': {}}

        # VECTORIZED: Pre-compute shared data structures
        shared_config = {
            'model_types': self.config.model_types,
            'enable_hpo': self.config.enable_hpo,
            'hpo_search_spaces': self.config.hpo_search_spaces
        }

        with ThreadPoolExecutor(max_workers=min(len(regime_data), 4)) as executor:
            future_to_regime = {}

            for regime, data in regime_data.items():
                future = executor.submit(
                    self._train_single_regime_vectorized,
                    regime, data, shared_config, feature_names
                )
                future_to_regime[future] = regime

            for future in as_completed(future_to_regime):
                regime = future_to_regime[future]
                try:
                    regime_result = future.result()
                    if regime_result:
                        results['models'][regime] = regime_result['models']
                        results['metadata'][regime] = regime_result['metadata']
                        self.logger.info(f"✅ VECTORIZED: Completed training for regime {regime}")
                except Exception as e:
                    self.logger.error(f"❌ VECTORIZED: Failed to train regime {regime}: {e}")

        return results

    def _train_small_regimes_batch(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Train small regimes with batch processing."""
        results = {'models': {}, 'metadata': {}}

        # VECTORIZED: Group by model type for batch training
        model_type_groups = {}

        for regime, data in regime_data.items():
            for model_type in self.config.model_types:
                if model_type not in model_type_groups:
                    model_type_groups[model_type] = []
                model_type_groups[model_type].append((regime, data))

        # VECTORIZED: Train each model type across all regimes simultaneously
        for model_type, regime_list in model_type_groups.items():
            self.logger.info(f"📊 VECTORIZED: Batch training {model_type} for {len(regime_list)} regimes")

            # VECTORIZED: Pre-compute shared training parameters
            if self.config.enable_hpo:
                search_space = self.config.hpo_search_spaces.get(model_type, {})
                batch_results = self._batch_hpo_training(model_type, regime_list, search_space, feature_names)
            else:
                batch_results = self._batch_standard_training(model_type, regime_list, feature_names)

            # VECTORIZED: Update results
            for regime, model_result in batch_results.items():
                if regime not in results['models']:
                    results['models'][regime] = {}
                    results['metadata'][regime] = {}

                results['models'][regime][model_type] = model_result['model']
                results['metadata'][regime].update(model_result['metadata'])

        return results

    def _batch_hpo_training(
        self,
        model_type: str,
        regime_list: List[Tuple[int, Dict[str, np.ndarray]]],
        search_space: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """VECTORIZED: Perform HPO across multiple regimes simultaneously."""

        results = {}

        with ThreadPoolExecutor(max_workers=min(len(regime_list), 4)) as executor:
            future_to_regime = {}

            for regime, data in regime_list:
                future = executor.submit(
                    self._optimize_single_regime_hpo,
                    model_type, regime, data, search_space, feature_names
                )
                future_to_regime[future] = regime

            for future in as_completed(future_to_regime):
                regime = future_to_regime[future]
                try:
                    result = future.result()
                    if result:
                        results[regime] = result
                        self.logger.info(f"✅ VECTORIZED HPO: Completed {model_type} for regime {regime}")
                except Exception as e:
                    self.logger.error(f"❌ VECTORIZED HPO: Failed {model_type} for regime {regime}: {e}")

        return results

    def _batch_standard_training(
        self,
        model_type: str,
        regime_list: List[Tuple[int, Dict[str, np.ndarray]]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """VECTORIZED: Train models across multiple regimes simultaneously."""

        results = {}

        with ThreadPoolExecutor(max_workers=min(len(regime_list), 4)) as executor:
            future_to_regime = {}

            for regime, data in regime_list:
                future = executor.submit(
                    self._train_single_regime_standard,
                    model_type, regime, data, feature_names
                )
                future_to_regime[future] = regime

            for future in as_completed(future_to_regime):
                regime = future_to_regime[future]
                try:
                    result = future.result()
                    if result:
                        results[regime] = result
                        self.logger.info(f"✅ VECTORIZED: Completed {model_type} for regime {regime}")
                except Exception as e:
                    self.logger.error(f"❌ VECTORIZED: Failed {model_type} for regime {regime}: {e}")

        return results

    def _train_single_regime_vectorized(
        self,
        regime: int,
        data: Dict[str, np.ndarray],
        config: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Train all model types for a single regime."""
        regime_models = {}
        regime_metadata = {
            'samples': data['samples'],
            'augmented': data['augmented'],
            'hmm_states': data.get('hmm_states'),
            'training_start': time.time()
        }

        for model_type in config['model_types']:
            try:
                self.logger.info(f"🔄 Training {model_type} for regime {regime}...")

                # VECTORIZED: HPO or standard training
                if config['enable_hpo']:
                    search_space = config['hpo_search_spaces'].get(model_type, {})
                    optimized_model = self.training_utils.optimize_model_with_hpo(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        search_space=search_space,
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )
                else:
                    optimized_model = self.training_utils.train_single_model(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )

                regime_models[model_type] = optimized_model

            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_type} for regime {regime}: {e}")

        regime_metadata['training_end'] = time.time()
        regime_metadata['training_time'] = regime_metadata['training_end'] - regime_metadata['training_start']

        return {
            'models': regime_models,
            'metadata': regime_metadata
        }

    def _optimize_single_regime_hpo(
        self,
        model_type: str,
        regime: int,
        data: Dict[str, np.ndarray],
        search_space: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Optimize hyperparameters for a single regime."""
        try:
            optimized_model = self.training_utils.optimize_model_with_hpo(
                model_type=model_type,
                X=data['X'],
                y=data['y'],
                search_space=search_space,
                model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
            )

            return {
                'model': optimized_model,
                'metadata': {
                    'model_type': model_type,
                    'regime': regime,
                    'hpo_performed': True,
                    'search_space': search_space,
                    'training_time': getattr(optimized_model, 'training_time', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"❌ HPO failed for {model_type} regime {regime}: {e}")
            return None

    def _train_single_regime_standard(
        self,
        model_type: str,
        regime: int,
        data: Dict[str, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Train a single model for a regime."""
        try:
            optimized_model = self.training_utils.train_single_model(
                model_type=model_type,
                X=data['X'],
                y=data['y'],
                model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
            )

            return {
                'model': optimized_model,
                'metadata': {
                    'model_type': model_type,
                    'regime': regime,
                    'hpo_performed': False,
                    'training_time': getattr(optimized_model, 'training_time', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Training failed for {model_type} regime {regime}: {e}")
            return None

    def execute_vectorized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Execute per-regime training step with ultra-fast batch processing.

        This method optimizes the entire training pipeline by:
        - Pre-computing regime analysis
        - Batch processing multiple regimes simultaneously
        - Parallel HPO across all regimes
        - Memory-efficient model saving and evaluation

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results and metadata
        """
        start_time = time.time()

        self.logger.info("🚀 VECTORIZED: Starting ultra-fast per-regime training...")
        self.logger.info(f"📊 Processing {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(regime_labels))} regimes")

        try:
            # VECTORIZED Step 1: Analyze regimes and prepare data
            self.logger.info("🔄 VECTORIZED Step 1: Analyzing regimes and preparing data...")
            regime_analysis = self.analyze_regimes(regime_labels)
            regime_data = self.prepare_regime_data(X, y, regime_labels, regime_analysis, hmm_states)

            # VECTORIZED Step 2: Train models for all regimes simultaneously
            self.logger.info("🔄 VECTORIZED Step 2: Training models for all regimes simultaneously...")
            regime_results = self.train_regime_models_vectorized(regime_data, feature_names)

            # VECTORIZED Step 3: Save models in parallel
            if self.config.save_models:
                self.logger.info("🔄 VECTORIZED Step 3: Saving trained models in parallel...")
                symbol = kwargs.get('symbol')
                exchange = kwargs.get('exchange')
                timeframe = kwargs.get('timeframe', self.config.timeframe)
                saved_paths = self.save_regime_models_vectorized(regime_results, symbol, exchange, timeframe)
            else:
                saved_paths = {}

            # VECTORIZED Step 4: Evaluate performance
            if self.config.enable_evaluation:
                self.logger.info("🔄 VECTORIZED Step 4: Evaluating model performance...")
                evaluation_results = self.evaluate_regime_models_vectorized(
                    regime_results, X, y, regime_labels,
                    is_classification=kwargs.get('is_classification', True)
                )
            else:
                evaluation_results = {}

            # VECTORIZED: Create final results
            total_time = time.time() - start_time
            results = self._create_final_results_vectorized(
                regime_results, regime_analysis, evaluation_results,
                saved_paths, total_time, **kwargs
            )

            self.logger.info(f"✅ VECTORIZED: Per-regime training completed in {total_time:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"❌ VECTORIZED per-regime training failed: {e}")
            # Fallback to standard method
            self.logger.info("🔄 Falling back to standard training method...")
            return self.execute(X, y, regime_labels, feature_names, hmm_states, **kwargs)

    def save_regime_models_vectorized(
        self,
        regime_results: Dict[str, Any],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Save trained models for all regimes simultaneously using parallel processing.

        Args:
            regime_results: Training results for each regime
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier

        Returns:
            Dictionary containing saved model paths for each regime
        """

        saved_paths = {}

        with ThreadPoolExecutor(max_workers=min(len(regime_results['models']), 4)) as executor:
            future_to_regime = {}

            for regime, models in regime_results['models'].items():
                # Extract models from results
                model_dict = {model_type: result['model'] for model_type, result in models.items()}

                future = executor.submit(
                    self._save_single_regime_models,
                    regime, model_dict, symbol, exchange, timeframe
                )
                future_to_regime[future] = regime

            for future in as_completed(future_to_regime):
                regime = future_to_regime[future]
                try:
                    regime_paths = future.result()
                    if regime_paths:
                        saved_paths[regime] = regime_paths
                        self.logger.info(f"💾 VECTORIZED: Saved models for regime {regime}")
                except Exception as e:
                    self.logger.error(f"❌ VECTORIZED: Failed to save models for regime {regime}: {e}")

        return saved_paths

    def _save_single_regime_models(
        self,
        regime: int,
        model_dict: Dict[str, Any],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """VECTORIZED: Save models for a single regime."""
        try:
            # Save models for this regime
            model_paths = self.save_models(
                models=model_dict,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )

            # Save regime metadata
            regime_metadata = {
                'regime': regime,
                'models_saved': list(model_paths.keys()),
                'save_timestamp': time.time()
            }

            self.save_metadata(
                metadata=regime_metadata,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )

            return model_paths

        except Exception as e:
            self.logger.error(f"❌ Failed to save models for regime {regime}: {e}")
            return {}

    def evaluate_regime_models_vectorized(
        self,
        regime_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Evaluate model performance for all regimes simultaneously.

        Args:
            regime_results: Training results for each regime
            X: Input features
            y: Target values
            regime_labels: Regime labels
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results for each regime
        """

        evaluation_results = {}

        with ThreadPoolExecutor(max_workers=min(len(regime_results['models']), 4)) as executor:
            future_to_regime = {}

            for regime, models in regime_results['models'].items():
                future = executor.submit(
                    self._evaluate_single_regime_models,
                    regime, models, X, y, regime_labels, is_classification
                )
                future_to_regime[future] = regime

            for future in as_completed(future_to_regime):
                regime = future_to_regime[future]
                try:
                    regime_eval = future.result()
                    if regime_eval:
                        evaluation_results[regime] = regime_eval
                        self.logger.info(f"📊 VECTORIZED: Evaluated models for regime {regime}")
                except Exception as e:
                    self.logger.error(f"❌ VECTORIZED: Failed to evaluate models for regime {regime}: {e}")

        return evaluation_results

    def _evaluate_single_regime_models(
        self,
        regime: int,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """VECTORIZED: Evaluate models for a single regime."""
        try:
            # Filter data for this regime
            regime_mask = regime_labels == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]

            if len(X_regime) == 0:
                return {}

            regime_eval = {}

            for model_type, model_result in models.items():
                try:
                    model = model_result['model']

                    # Generate predictions
                    predictions = model.predict(X_regime)

                    # Calculate evaluation metrics
                    if is_classification:
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        metrics = {
                            'accuracy': accuracy_score(y_regime, predictions),
                            'precision': precision_score(y_regime, predictions, average='weighted', zero_division=0),
                            'recall': recall_score(y_regime, predictions, average='weighted', zero_division=0),
                            'f1': f1_score(y_regime, predictions, average='weighted', zero_division=0)
                        }
                    else:
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        metrics = {
                            'mse': mean_squared_error(y_regime, predictions),
                            'mae': mean_absolute_error(y_regime, predictions),
                            'r2': r2_score(y_regime, predictions)
                        }

                    regime_eval[model_type] = {
                        'metrics': metrics,
                        'predictions': predictions,
                        'n_samples': len(X_regime)
                    }

                except Exception as e:
                    self.logger.error(f"❌ Failed to evaluate {model_type} for regime {regime}: {e}")

            return regime_eval

        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate regime {regime}: {e}")
            return {}

    def _create_final_results_vectorized(
        self,
        regime_results: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        saved_paths: Dict[str, Any],
        total_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """VECTORIZED: Create final results with comprehensive metadata."""
        results = {
            'models': regime_results['models'],
            'metadata': regime_results['metadata'],
            'regime_analysis': regime_analysis,
            'evaluation_results': evaluation_results,
            'saved_paths': saved_paths,
            'total_training_time': total_time,
            'vectorized': True,
            'config': {
                'model_name': self.config.model_name,
                'model_types': self.config.model_types,
                'enable_hpo': self.config.enable_hpo,
                'save_models': self.config.save_models,
                'enable_evaluation': self.config.enable_evaluation,
                'timeframe': self.config.timeframe
            },
            'summary': {
                'total_regimes': len(regime_results['models']),
                'total_models': sum(len(models) for models in regime_results['models'].values()),
                'total_saved_models': sum(len(paths) for paths in saved_paths.values()),
                'processing_method': 'vectorized'
            }
        }

        # Add kwargs to results
        for key, value in kwargs.items():
            if key not in results:
                results[key] = value

        return results

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute per-regime training step.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting per-regime training step")
        start_time = time.time()

        try:
            # Step 1: Analyze regimes and prepare data
            self.logger.info("🔄 Step 1: Analyzing regimes and preparing data...")
            regime_analysis = self.analyze_regimes(regime_labels)
            regime_data = self.prepare_regime_data(X, y, regime_labels, regime_analysis, hmm_states)

            # Step 2: Train models for each regime
            self.logger.info("🔄 Step 2: Training models for each regime...")
            regime_results = self.train_regime_models(regime_data, feature_names)

            # Step 3: Save models
            if self.config.save_models:
                self.logger.info("🔄 Step 3: Saving trained models...")
                symbol = kwargs.get('symbol')
                exchange = kwargs.get('exchange')
                timeframe = kwargs.get('timeframe', self.config.timeframe)
                self.save_regime_models(regime_results, symbol, exchange, timeframe)

            # Step 4: Evaluate performance
            if self.config.enable_evaluation:
                self.logger.info("🔄 Step 4: Evaluating model performance...")
                evaluation_results = self.evaluate_regime_models(
                    regime_results, X, y, regime_labels,
                    is_classification=kwargs.get('is_classification', True)
                )
            else:
                evaluation_results = {}

            # Create final results
            total_time = time.time() - start_time
            results = self._create_final_results(
                models=regime_results['models'],
                metadata=regime_results['metadata'],
                evaluation_results=evaluation_results,
                training_time=total_time,
                additional_results={'regime_analysis': regime_analysis}
            )

            self.training_results = results

            # Log summary
            n_models = sum(len(models) for models in regime_results['models'].values())
            self._log_training_summary(results, f"Per-regime {self.config.model_name}", n_models)

            return results

        except Exception as e:
            return self._handle_training_error(e, "per-regime training")
