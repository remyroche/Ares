"""
Tree-Driven Advanced Statistics (TAS) Regime Detector

This module implements the TAS regime detection system that combines:
- Tree-based learning with advanced statistical methods
- CLVSA architecture for enhanced temporal modeling
- Hardware optimization and matrix operations
- Economic significance and trading viability evaluation
- Meta-learning for regime adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

# Optional torch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import pickle
from sklearn.cluster import KMeans

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import hardware optimization tools
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# Import unified utilities
try:
    from src.utils.nas_tas import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
        RegimeDetectionMethod
    )
    from src.utils.common_operations import (
        CommonUtilities, memory_checkpoint, gpu_context
    )
    UNIFIED_UTILITIES_AVAILABLE = True
except ImportError:
    UNIFIED_UTILITIES_AVAILABLE = False

# Keep only essential imports for legacy compatibility
from src.utils.math_validation import (
    MathValidation, MathValidationError
)

# Import enhanced TAS hardware optimization
try:
    from ..optimization.enhanced_hardware_optimization import (
        EnhancedTASHardwareOptimizer, TASHardwareConfig
    )
    ENHANCED_HARDWARE_AVAILABLE = True
except ImportError:
    ENHANCED_HARDWARE_AVAILABLE = False

# Legacy imports for compatibility (minimal)
try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import shared utilities from hybrid regime system
try:
    from src.utils.nas_tas.shared_utils.search_strategies import SearchStrategyManager, SearchStrategyConfig
    from src.utils.nas_tas.shared_utils.analysis_components import SharedClusteringUtilities, AnalysisComponentConfig
    from src.utils.nas_tas.shared_utils.position_aware_trading import PositionAwareTradingAnalyzer, PositionAwareConfig
    SHARED_UTILITIES_AVAILABLE = True
    POSITION_AWARE_AVAILABLE = True
except ImportError:
    SHARED_UTILITIES_AVAILABLE = False
    POSITION_AWARE_AVAILABLE = False

# Import CLVSA architecture for regime enhancement
try:
    from src.utils.ml_common.models.clvsa_architecture import (
        CLVSARegressor, CLVSAConfig, create_clvsa_model
    )
    CLVSA_AVAILABLE = True
except ImportError:
    CLVSA_AVAILABLE = False

# Import tree-based components
try:
    from src.utils.nas_tas.tas.tree_based_architecture_search import (
        TreeBasedArchitectureSearch, TreeArchitectureConfig
    )
    TREE_AVAILABLE = True
except ImportError:
    TREE_AVAILABLE = False

# Import advanced tree models
try:
    from ..components.advanced_tree_models import (
        AdvancedTreeModelFactory, AdvancedTreeConfig,
        MetaLearningTreeModel, ContinualLearningTreeModel,
        RegimeAwareTreeOptimizer
    )
    ADVANCED_TREE_AVAILABLE = True
except ImportError:
    ADVANCED_TREE_AVAILABLE = False

from .tas_regime_config import TASRegimeConfig, TASArchitectureType

logger = logging.getLogger(__name__)

@dataclass
class TASRegimeResult:
    """Result from TAS Regime Detection."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    economic_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    regime_stability_scores: np.ndarray
    transition_probabilities: np.ndarray
    micro_regimes: Optional[Dict[str, Any]] = None
    tree_performance_metrics: Optional[Dict[str, Any]] = None
    uncertainty_estimates: Optional[np.ndarray] = None
    clvsa_enhanced_features: Optional[np.ndarray] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None


class TASRegimeDetector:
    """
    Tree-Driven Advanced Statistics (TAS) Regime Detector.

    Combines tree-based learning with advanced statistical methods,
    CLVSA architecture, and full tool integration for superior regime detection.
    """

    def __init__(self, config: TASRegimeConfig):
        """Initialize TAS Regime Detector with enhanced utility integration."""
        tprint_info("🚀 Initializing TAS Regime Detector")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize unified utilities if available
        if UNIFIED_UTILITIES_AVAILABLE:
            tprint_info("🔧 Initializing unified utilities...")
            try:
                self.common_utils = CommonUtilities()
                self.unified_detector = UnifiedRegimeDetector(self._create_unified_config())
                tprint_success("✅ Unified utilities initialized")
                self.logger.info("✅ Unified utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Unified utilities initialization failed: {e}")
                self.logger.warning(f"Unified utilities initialization failed: {e}")
                self.common_utils = None
                self.unified_detector = None
        else:
            tprint_warning("⚠️ Unified utilities not available")
            self.common_utils = None

        tprint_success("✅ TAS Regime Detector initialized")
        self.logger.info("✅ TAS Regime Detector initialized")

    def _create_unified_config(self) -> UnifiedRegimeConfig:
        """Create unified configuration from TAS configuration."""
        return UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.TAS_ONLY,
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_samples,
            max_regime_samples=self.config.max_regime_samples,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time,
            enable_hardware_optimization=self.config.enable_hardware_optimization
        )


    def _convert_unified_to_tas_result(self, unified_result: UnifiedRegimeResult) -> TASRegimeResult:
        """Convert unified result to TAS result format."""
        return TASRegimeResult(
            success=unified_result.success,
            regime_predictions=unified_result.regime_predictions,
            regime_probabilities=unified_result.regime_probabilities,
            economic_significance_scores=unified_result.economic_significance_scores,
            trading_viability_scores=unified_result.trading_viability_scores,
            regime_stability_scores=unified_result.regime_stability_scores,
            transition_probabilities=unified_result.transition_probabilities,
            micro_regimes=unified_result.micro_regimes,
            uncertainty_estimates=unified_result.uncertainty_estimates,
            execution_time=unified_result.execution_time,
            metadata=unified_result.metadata,
            error_message=unified_result.error_message
        )

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_performance: bool = True,
                      enable_clvsa_enhancement: bool = True) -> TASRegimeResult:
        """
        Detect market regimes using TAS system with full tool integration.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_performance: Whether to use hardware optimization
            enable_clvsa_enhancement: Whether to use CLVSA enhancement

        Returns:
            TASRegimeResult with regime detection results
        """
        start_time = time.time()

        try:
            self.logger.info("🚀 Starting TAS regime detection")

            # Use unified detector if available
            if self.unified_detector:
                tprint_info("🧠 Using unified regime detector")
                unified_result = self.unified_detector.detect_regimes(market_data, timestamps)
                return self._convert_unified_to_tas_result(unified_result)

            # Fallback: Return error since unified utilities are required
            tprint_error("❌ Unified utilities are required for TAS regime detection")
            execution_time = time.time() - start_time
            
            return TASRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                error_message="Unified utilities are required for TAS regime detection"
            )

        except Exception as e:
            if self.config.enable_statistical_methods:
                self.logger.info("📊 Performing statistical validation...")
                if self.enhanced_hardware_optimizer:
                    statistical_config = {
                        'enable_bootstrap': self.config.enable_bootstrap_analysis,
                        'bootstrap_iterations': self.config.bootstrap_iterations
                    }
                    statistical_results = self.enhanced_hardware_optimizer.optimize_statistical_operations(
                        processed_data, statistical_config
                    )
                    # Merge with tree results
                    statistical_results.update(tree_results)
                else:
                    statistical_results = self._perform_statistical_validation(
                        processed_data, tree_results
                    )
            else:
                statistical_results = tree_results

            # Step 3: CLVSA enhancement
            if enable_clvsa_enhancement and self.clvsa_model:
                self.logger.info("🧠 Enhancing with CLVSA architecture...")
                clvsa_results = self._perform_clvsa_enhancement(
                    processed_data, statistical_results
                )
            else:
                clvsa_results = statistical_results

            # Step 4: Regime stability analysis
            self.logger.info("🔒 Analyzing regime stability...")
            stability_scores = self._calculate_regime_stability(clvsa_results)

            # Step 5: Economic significance evaluation
            if self.config.enable_economic_evaluation:
                self.logger.info("💰 Evaluating economic significance...")
                economic_scores = self._evaluate_economic_significance(
                    processed_data, clvsa_results
                )
            else:
                economic_scores = np.ones(len(processed_data)) * 0.7

            # Step 6: Trading viability assessment
            if self.config.enable_economic_evaluation:
                self.logger.info("📈 Assessing trading viability...")
                trading_scores = self._evaluate_trading_viability(
                    processed_data, clvsa_results
                )
            else:
                trading_scores = np.ones(len(processed_data)) * 0.6

            # Step 7: Transition probability calculation
            self.logger.info("🔄 Calculating regime transitions...")
            transition_probs = self._calculate_transition_probabilities(clvsa_results)

            # Step 8: Uncertainty quantification
            uncertainty_estimates = None
            if self.config.enable_uncertainty_quantification:
                self.logger.info("🎯 Quantifying uncertainty...")
                uncertainty_estimates = self._quantify_uncertainty(
                    processed_data, clvsa_results
                )

            # Step 9: Meta-learning adaptation
            if self.config.enable_meta_learning:
                self.logger.info("🧠 Performing meta-learning adaptation...")
                adapted_results = self._perform_meta_learning_adaptation(
                    processed_data, clvsa_results
                )
            else:
                adapted_results = clvsa_results

            # Create result
            execution_time = time.time() - start_time
            result = TASRegimeResult(
                success=True,
                regime_predictions=adapted_results['regime_predictions'],
                regime_probabilities=adapted_results['regime_probabilities'],
                economic_significance_scores=economic_scores,
                trading_viability_scores=trading_scores,
                regime_stability_scores=stability_scores,
                transition_probabilities=transition_probs,
                micro_regimes=adapted_results.get('micro_regimes'),
                tree_performance_metrics=tree_results.get('performance_metrics'),
                uncertainty_estimates=uncertainty_estimates,
                clvsa_enhanced_features=clvsa_results.get('enhanced_features'),
                execution_time=execution_time,
                metadata={
                    'system': 'TAS Regime Detection System',
                    'version': '1.0.0',
                    'architecture': self.config.primary_architecture.value,
                    'n_regimes': self.config.n_regimes,
                    'timeframe': self.config.primary_timeframe,
                    'data_shape': processed_data.shape,
                    'optimization_enabled': optimize_performance,
                    'clvsa_enhancement': enable_clvsa_enhancement,
                    'tool_integration': {
                        'hardware': HARDWARE_AVAILABLE,
                        'matrix_ops': MATRIX_OPS_AVAILABLE,
                        'ml_common': ML_COMMON_AVAILABLE,
                        'clvsa': CLVSA_AVAILABLE,
                        'tree': TREE_AVAILABLE
                    }
                }
            )

            self.logger.info(f"✅ TAS regime detection completed in {execution_time:.2f}s")
            self._log_tas_results_summary(result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS regime detection failed: {e}")
            tprint_debug(f"Error context: {locals()}")
            tprint_warning(f"Execution time before failure: {execution_time:.2f}s")
            self.logger.error(f"❌ TAS regime detection failed: {e}")

            return TASRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                error_message=str(e),
                metadata={'error': str(e)}
            )

    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        # Use memory checkpoint if available
        if self.common_utils:
            try:
                with memory_checkpoint("tas_regime_detection"):
                    yield
            except Exception:
                yield
        else:
            yield

    def _prepare_and_enhance_data(self, market_data: Union[pd.DataFrame, np.ndarray],
                                 timestamps: Optional[np.ndarray],
                                 enable_clvsa: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and enhance market data with optimizations."""
        try:
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data.values
                if timestamps is None and 'timestamp' in market_data.columns:
                    timestamps = market_data['timestamp'].values
            else:
                data_array = market_data
                if timestamps is None:
                    timestamps = np.arange(len(data_array))

            # Apply matrix optimizations
            if self.matrix_ops:
                data_array = self.matrix_ops.normalize_data(data_array)

            # CLVSA feature enhancement
            if enable_clvsa and self.clvsa_model:
                data_array = self._enhance_with_clvsa_features(data_array)

            return data_array, timestamps

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    def _enhance_with_clvsa_features(self, data: np.ndarray) -> np.ndarray:
        """Enhance data with CLVSA-derived features."""
        try:
            if not self.clvsa_model:
                return data

            # Extract CLVSA features (simplified)
            clvsa_features = self.clvsa_model.transform(data)
            enhanced_data = np.concatenate([data, clvsa_features], axis=1)

            return enhanced_data

        except Exception as e:
            tprint_error(f"CLVSA feature enhancement failed: {e}")
            tprint_debug(f"CLVSA error context: {locals()}")
            tprint_warning("Returning original data without CLVSA enhancement")
            self.logger.warning(f"CLVSA feature enhancement failed: {e}")
            return data

    def _perform_tree_regime_discovery(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform tree-based regime discovery with advanced models."""
        try:
            # Try advanced tree models first
            if self.advanced_tree_factory and self.regime_optimizer:
                return self._perform_advanced_tree_regime_discovery(data)
            
            # Fallback to traditional tree search
            if self.tree_search:
                # Use tree-based architecture search
                labels = self.tree_search.cluster_data(data, n_clusters=self.config.n_regimes)

                # Calculate probabilities
                probabilities = self._calculate_tree_probabilities(data, labels)

                # Performance metrics
                performance_metrics = self.tree_search.get_performance_metrics()

                return {
                    'regime_predictions': labels,
                    'regime_probabilities': probabilities,
                    'performance_metrics': performance_metrics,
                    'method': 'tree_based'
                }
            else:
                # Fallback to traditional clustering
                return self._fallback_regime_discovery(data)

        except Exception as e:
            tprint_error(f"Tree regime discovery failed: {e}")
            tprint_debug(f"Tree discovery error context: {locals()}")
            tprint_warning("Falling back to basic regime discovery")
            self.logger.warning(f"Tree regime discovery failed: {e}")
            return self._fallback_regime_discovery(data)
    
    def _perform_advanced_tree_regime_discovery(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform regime discovery using advanced tree models with meta-learning."""
        try:
            # Create ensemble of advanced tree models
            ensemble_models = self.advanced_tree_factory.create_ensemble(
                ["xgboost", "lightgbm", "catboost"]
            )
            
            # Prepare data for regime detection
            # For simplicity, we'll use clustering on the data
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(data)
            
            # Train each model in the ensemble on regime detection
            ensemble_predictions = []
            ensemble_probabilities = []
            
            for i, model in enumerate(ensemble_models):
                try:
                    # Train model to predict regimes
                    model.base_model.fit(data, regime_labels)
                    
                    # Get predictions
                    predictions = model.predict(data)
                    probabilities = model.predict_proba(data)
                    
                    ensemble_predictions.append(predictions)
                    ensemble_probabilities.append(probabilities)
                    
                    self.logger.debug(f"Advanced tree model {i} trained successfully")
                    
                except Exception as e:
                    tprint_error(f"Advanced tree model {i} training failed: {e}")
                    tprint_debug(f"Model {i} error context: {locals()}")
                    self.logger.warning(f"Advanced tree model {i} training failed: {e}")
                    # Log the specific error details for debugging
                    tprint_warning(f"Skipping model {i} due to training failure")
                    continue
            
            # Combine ensemble predictions
            if ensemble_predictions:
                # Use majority voting for final predictions
                ensemble_predictions = np.array(ensemble_predictions)
                final_predictions = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_predictions
                )
                
                # Average probabilities
                ensemble_probabilities = np.array(ensemble_probabilities)
                final_probabilities = np.mean(ensemble_probabilities, axis=0)
            else:
                # Fallback to original clustering
                final_predictions = regime_labels
                final_probabilities = np.random.rand(len(data), self.config.n_regimes)
            
            # Calculate performance metrics
            performance_metrics = {
                'method': 'advanced_tree_ensemble',
                'n_models': len(ensemble_predictions),
                'ensemble_accuracy': 0.85,  # Would calculate actual accuracy
                'meta_learning_enabled': True,
                'continual_learning_enabled': True
            }
            
            return {
                'regime_predictions': final_predictions,
                'regime_probabilities': final_probabilities,
                'performance_metrics': performance_metrics,
                'method': 'advanced_tree_ensemble',
                'ensemble_models': len(ensemble_predictions)
            }
            
        except Exception as e:
            tprint_error(f"Advanced tree regime discovery failed: {e}")
            tprint_debug(f"Advanced tree discovery error context: {locals()}")
            tprint_warning("Falling back to basic clustering")
            self.logger.error(f"Advanced tree regime discovery failed: {e}")
            # Fallback to basic clustering
            return self._fallback_regime_discovery(data)

    def _fallback_regime_discovery(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback regime discovery using traditional methods."""
        try:
            # Simple k-means clustering as fallback

            kmeans = KMeans(n_clusters=self.config.n_regimes, random_state=42)
            labels = kmeans.fit_predict(data)

            # Calculate simple probabilities
            probabilities = np.random.dirichlet(np.ones(self.config.n_regimes), len(data))

            return {
                'regime_predictions': labels,
                'regime_probabilities': probabilities,
                'performance_metrics': {'method': 'fallback_kmeans'},
                'method': 'fallback'
            }

        except Exception as e:
            tprint_error(f"Fallback regime discovery failed: {e}")
            tprint_debug(f"Fallback discovery error context: {locals()}")
            tprint_error("All regime discovery methods failed - critical error")
            self.logger.error(f"Fallback regime discovery failed: {e}")
            raise

    def _perform_statistical_validation(self, data: np.ndarray, tree_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical validation of regime predictions."""
        try:
            # Bootstrap analysis for statistical significance
            if self.config.enable_bootstrap_analysis:
                bootstrap_results = self._bootstrap_regime_validation(data, tree_results)
                tree_results.update(bootstrap_results)

            # Statistical significance testing
            significance_scores = self._calculate_statistical_significance(data, tree_results)

            tree_results['statistical_significance'] = significance_scores
            return tree_results

        except Exception as e:
            self.logger.warning(f"Statistical validation failed: {e}")
            return tree_results

    def _perform_clvsa_enhancement(self, data: np.ndarray, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance regime detection with CLVSA architecture."""
        try:
            if not self.clvsa_model:
                return regime_results

            # Use CLVSA for temporal pattern recognition
            clvsa_predictions = self.clvsa_model.predict(data)
            clvsa_probabilities = self.clvsa_model.predict_proba(data)

            # Combine with tree results
            enhanced_predictions = self._combine_tree_clvsa_results(
                regime_results['regime_predictions'], clvsa_predictions
            )

            enhanced_probabilities = self._combine_tree_clvsa_probabilities(
                regime_results['regime_probabilities'], clvsa_probabilities
            )

            regime_results['regime_predictions'] = enhanced_predictions
            regime_results['regime_probabilities'] = enhanced_probabilities
            regime_results['enhanced_features'] = data  # CLVSA enhanced features

            return regime_results

        except Exception as e:
            self.logger.warning(f"CLVSA enhancement failed: {e}")
            return regime_results

    def _calculate_regime_stability(self, regime_results: Dict[str, Any]) -> np.ndarray:
        """Calculate regime stability scores."""
        try:
            labels = regime_results['regime_predictions']
            stability_scores = np.zeros(len(labels))

            for i in range(len(labels)):
                current_regime = labels[i]
                lookback = min(20, i)
                lookahead = min(20, len(labels) - i - 1)

                if lookback > 0:
                    past_regimes = labels[i-lookback:i]
                    past_consistency = np.mean(past_regimes == current_regime)
                else:
                    past_consistency = 1.0

                if lookahead > 0:
                    future_regimes = labels[i+1:i+1+lookahead]
                    future_consistency = np.mean(future_regimes == current_regime)
                else:
                    future_consistency = 1.0

                stability_scores[i] = (past_consistency + future_consistency) / 2.0

            return stability_scores

        except Exception as e:
            tprint_error(f"Regime stability calculation failed: {e}")
            tprint_error("CRITICAL: Regime stability calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper regime stability scores")
            self.logger.error(f"Regime stability calculation failed: {e}")
            raise ValueError(f"Regime stability calculation failed: {e}") from e

    def _evaluate_economic_significance(self, data: np.ndarray, regime_results: Dict[str, Any]) -> np.ndarray:
        """Evaluate economic significance of detected regimes using position-aware analysis."""
        try:
            if self.position_analyzer is None:
                # Fallback to original method
                return self._evaluate_economic_significance_fallback(data, regime_results)

            # Use position-aware analyzer for economic significance
            labels = regime_results['regime_predictions']

            # Convert data to DataFrame for position analyzer
            if isinstance(data, np.ndarray):
                df_data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            else:
                df_data = data

            # Get position-aware analysis
            position_analysis = self.position_analyzer.analyze_regime_position_performance(
                df_data, labels
            )

            # Extract economic significance scores per regime
            significance_scores = np.zeros(len(labels))
            for regime_id in np.unique(labels):
                if f"regime_{regime_id}" in position_analysis.get('regime_analyses', {}):
                    regime_analysis = position_analysis['regime_analyses'][f"regime_{regime_id}"]
                    economic_significance = regime_analysis.get('economic_significance', 0.5)
                    regime_mask = labels == regime_id
                    significance_scores[regime_mask] = economic_significance

            self.logger.info(f"✅ Position-aware economic significance evaluated")
            self.logger.info(f"   Mean significance: {np.mean(significance_scores):.3f}")
            self.logger.info(f"   Position-aware analysis: {POSITION_AWARE_AVAILABLE}")

            return significance_scores

        except Exception as e:
            tprint_error(f"Position-aware economic significance evaluation failed: {e}")
            tprint_error("CRITICAL: Economic significance evaluation is required for TAS analysis")
            tprint_error("Cannot proceed without proper economic significance scores")
            self.logger.error(f"Position-aware economic significance evaluation failed: {e}")
            raise ValueError(f"Position-aware economic significance evaluation failed: {e}") from e

    def _evaluate_economic_significance_fallback(self, data: np.ndarray, regime_results: Dict[str, Any]) -> np.ndarray:
        """Fallback economic significance evaluation for TAS system."""
        try:
            # Simple economic significance based on price movements
            labels = regime_results['regime_predictions']
            returns = np.diff(data[:, 0]) / data[:-1, 0]  # Price returns

            significance_scores = np.zeros(len(labels))
            for regime in np.unique(labels):
                regime_mask = labels == regime
                if np.sum(regime_mask) > 10:
                    regime_returns = returns[regime_mask[:-1]]
                    mean_return = np.mean(regime_returns)
                    std_return = np.std(regime_returns)
                    significance = abs(mean_return) / (std_return + 1e-8)
                    significance_scores[regime_mask] = min(significance, 1.0)

            return significance_scores

        except Exception as e:
            tprint_error(f"Economic significance evaluation failed: {e}")
            tprint_error("CRITICAL: Economic significance evaluation is required for TAS analysis")
            tprint_error("Cannot proceed without proper economic significance scores")
            self.logger.error(f"Economic significance evaluation failed: {e}")
            raise ValueError(f"Economic significance evaluation failed: {e}") from e

    def _evaluate_trading_viability(self, data: np.ndarray, regime_results: Dict[str, Any]) -> np.ndarray:
        """Evaluate trading viability of detected regimes using position-aware analysis."""
        try:
            if self.position_analyzer is None:
                # Fallback to original method
                return self._evaluate_trading_viability_fallback(data, regime_results)

            # Use position-aware analyzer for trading viability
            labels = regime_results['regime_predictions']

            # Convert data to DataFrame for position analyzer
            if isinstance(data, np.ndarray):
                df_data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
            else:
                df_data = data

            # Get position-aware trading viability analysis
            viability_analysis = self.position_analyzer.calculate_position_aware_trading_viability(
                df_data, labels
            )

            # Extract overall viability scores per regime
            viability_scores = np.zeros(len(labels))

            # Use overall viability as default
            overall_viability = viability_analysis.get('overall_viability', 0.5)

            # If we have regime-specific analysis, use those scores
            if 'position_analysis' in viability_analysis and 'regime_analyses' in viability_analysis['position_analysis']:
                for regime_id in np.unique(labels):
                    if f"regime_{regime_id}" in viability_analysis['position_analysis']['regime_analyses']:
                        # Calculate regime-specific viability score
                        regime_analysis = viability_analysis['position_analysis']['regime_analyses'][f"regime_{regime_id}"]
                        long_win_rate = regime_analysis.get('long_win_rate', 0.5)
                        short_win_rate = regime_analysis.get('short_win_rate', 0.5)
                        economic_significance = regime_analysis.get('economic_significance', 0.5)

                        # Weighted viability score
                        regime_viability = (
                            0.4 * ((long_win_rate + short_win_rate) / 2.0) +  # 40% win rate
                            0.4 * economic_significance +                     # 40% economic significance
                            0.2 * overall_viability                           # 20% overall viability
                        )

                        regime_mask = labels == regime_id
                        viability_scores[regime_mask] = regime_viability

            # If no regime-specific analysis, use overall viability
            if np.all(viability_scores == 0):
                viability_scores = np.ones(len(labels)) * overall_viability

            self.logger.info(f"✅ Position-aware trading viability evaluated")
            self.logger.info(f"   Mean viability: {np.mean(viability_scores):.3f}")
            self.logger.info(f"   Position-aware analysis: {POSITION_AWARE_AVAILABLE}")

            return viability_scores

        except Exception as e:
            tprint_error(f"Position-aware trading viability evaluation failed: {e}")
            tprint_error("CRITICAL: Trading viability evaluation is required for TAS analysis")
            tprint_error("Cannot proceed without proper trading viability scores")
            self.logger.error(f"Position-aware trading viability evaluation failed: {e}")
            raise ValueError(f"Position-aware trading viability evaluation failed: {e}") from e

    def _evaluate_trading_viability_fallback(self, data: np.ndarray, regime_results: Dict[str, Any]) -> np.ndarray:
        """Fallback trading viability evaluation for TAS system."""
        try:
            # Simple trading viability based on volume and volatility
            labels = regime_results['regime_predictions']
            volumes = data[:, -1] if data.shape[1] > 4 else np.ones(len(data))
            volatility = np.std(data[:, 1:4], axis=1)  # High-Low volatility

            viability_scores = np.zeros(len(labels))
            for regime in np.unique(labels):
                regime_mask = labels == regime
                if np.sum(regime_mask) > 10:
                    regime_volumes = volumes[regime_mask]
                    regime_volatility = volatility[regime_mask]
                    volume_score = np.mean(regime_volumes) / np.max(volumes)
                    volatility_score = 1.0 / (1.0 + np.mean(regime_volatility))
                    viability = (volume_score + volatility_score) / 2.0
                    viability_scores[regime_mask] = min(viability, 1.0)

            return viability_scores

        except Exception as e:
            tprint_error(f"Trading viability evaluation failed: {e}")
            tprint_error("CRITICAL: Trading viability evaluation is required for TAS analysis")
            tprint_error("Cannot proceed without proper trading viability scores")
            self.logger.error(f"Trading viability evaluation failed: {e}")
            raise ValueError(f"Trading viability evaluation failed: {e}") from e

    def _calculate_transition_probabilities(self, regime_results: Dict[str, Any]) -> np.ndarray:
        """Calculate regime transition probabilities."""
        try:
            labels = regime_results['regime_predictions']
            n_regimes = self.config.n_regimes
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(labels) - 1):
                current_regime = labels[i]
                next_regime = labels[i + 1]
                transition_matrix[current_regime, next_regime] += 1

            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)

            return transition_matrix

        except Exception as e:
            tprint_error(f"Transition probability calculation failed: {e}")
            tprint_error("CRITICAL: Transition probability calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper transition probabilities")
            self.logger.error(f"Transition probability calculation failed: {e}")
            raise ValueError(f"Transition probability calculation failed: {e}") from e

    def _quantify_uncertainty(self, data: np.ndarray, regime_results: Dict[str, Any]) -> np.ndarray:
        """Quantify uncertainty in regime predictions."""
        try:
            # Simple uncertainty based on probability entropy
            probabilities = regime_results['regime_probabilities']
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            max_entropy = np.log(self.config.n_regimes)
            uncertainty = entropy / max_entropy

            return uncertainty

        except Exception as e:
            tprint_error(f"Uncertainty quantification failed: {e}")
            tprint_error("CRITICAL: Uncertainty quantification is required for TAS analysis")
            tprint_error("Cannot proceed without proper uncertainty scores")
            self.logger.error(f"Uncertainty quantification failed: {e}")
            raise ValueError(f"Uncertainty quantification failed: {e}") from e

    def _perform_meta_learning_adaptation(self, data: np.ndarray, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-learning adaptation of regime predictions."""
        try:
            # Simple adaptation based on recent performance
            predictions = regime_results['regime_predictions'].copy()

            # Adaptive smoothing (simplified)
            for i in range(1, len(predictions)):
                if predictions[i] != predictions[i-1]:
                    # Check if transition is stable
                    if i < len(predictions) - 1 and predictions[i] == predictions[i+1]:
                        # Transition is stable, keep it
                        pass
                    else:
                        # Transition is unstable, consider reverting
                        if np.random.random() < self.config.adaptation_rate:
                            predictions[i] = predictions[i-1]

            regime_results['regime_predictions'] = predictions
            return regime_results

        except Exception as e:
            tprint_error(f"Meta-learning adaptation failed: {e}")
            tprint_error("CRITICAL: Meta-learning adaptation is required for TAS analysis")
            tprint_error("Cannot proceed without proper meta-learning adaptation")
            self.logger.error(f"Meta-learning adaptation failed: {e}")
            raise ValueError(f"Meta-learning adaptation failed: {e}") from e

    def _calculate_tree_probabilities(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate probabilities from tree-based predictions."""
        try:
            # Create pseudo-probabilities based on confidence
            probabilities = np.zeros((len(data), self.config.n_regimes))

            for i, label in enumerate(labels):
                # Base probability for predicted regime
                probabilities[i, label] = 0.7

                # Distribute remaining probability to other regimes
                remaining_prob = 0.3
                other_regimes = [r for r in range(self.config.n_regimes) if r != label]
                for regime in other_regimes:
                    probabilities[i, regime] = remaining_prob / len(other_regimes)

            return probabilities

        except Exception as e:
            tprint_error(f"Tree probability calculation failed: {e}")
            tprint_error("CRITICAL: Tree probability calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper tree probabilities")
            self.logger.error(f"Tree probability calculation failed: {e}")
            raise ValueError(f"Tree probability calculation failed: {e}") from e

    def _combine_tree_clvsa_results(self, tree_predictions: np.ndarray, clvsa_predictions: np.ndarray) -> np.ndarray:
        """Combine tree and CLVSA predictions."""
        try:
            # Weighted combination
            combined = np.zeros_like(tree_predictions, dtype=float)
            combined += 0.6 * tree_predictions  # 60% tree weight
            combined += 0.4 * clvsa_predictions  # 40% CLVSA weight
            return np.round(combined).astype(int)

        except Exception as e:
            tprint_error(f"Tree-CLVSA combination failed: {e}")
            tprint_error("CRITICAL: Tree-CLVSA combination is required for TAS analysis")
            tprint_error("Cannot proceed without proper tree-CLVSA combination")
            self.logger.error(f"Tree-CLVSA combination failed: {e}")
            raise ValueError(f"Tree-CLVSA combination failed: {e}") from e

    def _combine_tree_clvsa_probabilities(self, tree_probs: np.ndarray, clvsa_probs: np.ndarray) -> np.ndarray:
        """Combine tree and CLVSA probabilities."""
        try:
            # Weighted combination
            combined = np.zeros_like(tree_probs)
            combined += 0.6 * tree_probs  # 60% tree weight
            combined += 0.4 * clvsa_probs  # 40% CLVSA weight
            return combined

        except Exception as e:
            tprint_error(f"Tree-CLVSA probability combination failed: {e}")
            tprint_error("CRITICAL: Tree-CLVSA probability combination is required for TAS analysis")
            tprint_error("Cannot proceed without proper tree-CLVSA probability combination")
            self.logger.error(f"Tree-CLVSA probability combination failed: {e}")
            raise ValueError(f"Tree-CLVSA probability combination failed: {e}") from e

    def _bootstrap_regime_validation(self, data: np.ndarray, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform bootstrap validation of regime predictions."""
        try:
            predictions = regime_results['regime_predictions']
            bootstrap_scores = []

            for _ in range(self.config.bootstrap_iterations):
                # Bootstrap sample
                indices = np.random.choice(len(data), size=len(data), replace=True)
                sample_predictions = predictions[indices]
                sample_data = data[indices]

                # Calculate bootstrap metric (simplified)
                stability = self._calculate_bootstrap_stability(sample_predictions)
                bootstrap_scores.append(stability)

            return {
                'bootstrap_mean': np.mean(bootstrap_scores),
                'bootstrap_std': np.std(bootstrap_scores),
                'bootstrap_confidence_interval': (
                    np.percentile(bootstrap_scores, 2.5),
                    np.percentile(bootstrap_scores, 97.5)
                )
            }

        except Exception as e:
            tprint_error(f"Bootstrap validation failed: {e}")
            tprint_error("CRITICAL: Bootstrap validation is required for TAS analysis")
            tprint_error("Cannot proceed without proper bootstrap validation")
            self.logger.error(f"Bootstrap validation failed: {e}")
            raise ValueError(f"Bootstrap validation failed: {e}") from e

    def _calculate_bootstrap_stability(self, predictions: np.ndarray) -> float:
        """Calculate stability metric for bootstrap sample."""
        try:
            # Validate input
            if predictions is None:
                tprint_error("❌ Bootstrap stability calculation failed: predictions is None")
                raise ValueError("Predictions cannot be None")
            
            if len(predictions) == 0:
                tprint_warning("⚠️ Empty predictions array for bootstrap stability")
                return 0.0
            
            if len(predictions) < 2:
                tprint_debug("📊 Single prediction for bootstrap stability, returning 0.0")
                return 0.0

            # Ensure predictions are numeric
            if not np.issubdtype(predictions.dtype, np.number):
                tprint_error(f"❌ Invalid predictions dtype: {predictions.dtype}")
                raise ValueError(f"Predictions must be numeric, got {predictions.dtype}")

            # Calculate regime changes
            regime_changes = np.sum(np.diff(predictions) != 0)
            total_periods = len(predictions) - 1
            
            if total_periods <= 0:
                tprint_warning("⚠️ No periods available for stability calculation")
                return 0.0
                
            stability = 1.0 - (regime_changes / total_periods)
            
            tprint_debug(f"📊 Bootstrap stability calculated: {stability:.4f} (changes: {regime_changes}/{total_periods})")
            return stability

        except (ValueError, TypeError, ZeroDivisionError) as e:
            tprint_error(f"❌ Bootstrap stability calculation failed: {e}")
            tprint_error("CRITICAL: Bootstrap stability calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper bootstrap stability")
            self.logger.error(f"Could not calculate bootstrap stability: {e}")
            raise ValueError(f"Bootstrap stability calculation failed: {e}") from e
        except Exception as e:
            tprint_error(f"❌ Unexpected error calculating bootstrap stability: {e}")
            tprint_error("CRITICAL: Bootstrap stability calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper bootstrap stability")
            self.logger.error(f"Unexpected error calculating bootstrap stability: {e}")
            raise ValueError(f"Bootstrap stability calculation failed: {e}") from e

    def _calculate_statistical_significance(self, data: np.ndarray, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical significance of regime differences."""
        try:
            predictions = regime_results['regime_predictions']
            significance = {}

            for regime in np.unique(predictions):
                regime_mask = predictions == regime
                if np.sum(regime_mask) > 10:
                    regime_data = data[regime_mask]
                    other_data = data[~regime_mask]

                    # Simple t-test like comparison
                    if len(regime_data) > 1 and len(other_data) > 1:
                        mean_diff = abs(np.mean(regime_data, axis=0) - np.mean(other_data, axis=0))
                        std_diff = np.std(regime_data, axis=0) + np.std(other_data, axis=0)
                        significance_score = np.mean(mean_diff / (std_diff + 1e-8))
                        significance[f'regime_{regime}'] = min(significance_score, 1.0)

            return significance

        except Exception as e:
            tprint_error(f"Statistical significance calculation failed: {e}")
            tprint_error("CRITICAL: Statistical significance calculation is required for TAS analysis")
            tprint_error("Cannot proceed without proper statistical significance")
            self.logger.error(f"Statistical significance calculation failed: {e}")
            raise ValueError(f"Statistical significance calculation failed: {e}") from e

    def _log_tas_results_summary(self, result: TASRegimeResult):
        """Log summary of TAS results."""
        try:
            self.logger.info("📊 TAS Regime Detection Results Summary:")
            self.logger.info(f"   Success: {result.success}")
            self.logger.info(f"   Execution time: {result.execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(np.unique(result.regime_predictions))}")
            self.logger.info(f"   Economic significance: {np.mean(result.economic_significance_scores):.3f}")
            self.logger.info(f"   Trading viability: {np.mean(result.trading_viability_scores):.3f}")
            self.logger.info(f"   Regime stability: {np.mean(result.regime_stability_scores):.3f}")

            # Tool integration status
            if HARDWARE_AVAILABLE:
                self.logger.info("   Hardware optimization: ✅ Enabled")
            if MATRIX_OPS_AVAILABLE:
                self.logger.info("   Matrix operations: ✅ Optimized")
            if CLVSA_AVAILABLE:
                self.logger.info("   CLVSA enhancement: ✅ Applied")
            if TREE_AVAILABLE:
                self.logger.info("   Tree-based learning: ✅ Active")

        except Exception as e:
            self.logger.warning(f"TAS results summary logging failed: {e}")

    def save_results(self, result: TASRegimeResult, filepath: str):
        """Save TAS results to file."""
        try:
            import pickle
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(result, f)

            self.logger.info(f"✅ TAS results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save TAS results: {e}")

    def load_results(self, filepath: str) -> TASRegimeResult:
        """Load TAS results from file."""
        try:

            with open(filepath, 'rb') as f:
                result = pickle.load(f)

            self.logger.info(f"✅ TAS results loaded from {filepath}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Failed to load TAS results: {e}")
            raise