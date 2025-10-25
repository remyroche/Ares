"""
Feature Service for HDBSCAN Clustering.

This module provides feature preparation, scaling, and embedding services
that wrap FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Mac M1 Hardware Optimizations
HARDWARE_OPTIMIZATIONS_AVAILABLE = False
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    pass

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import (
    get_logger,
    prepare_market_features,
    FeatureConfig
)

# Import utility functions
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes
)
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power
)

@dataclass
class FeaturePreparationResult:
    """Result from feature preparation."""
    features: np.ndarray
    feature_names: List[str]
    feature_scores: Dict[str, float]
    dropped_features: List[str]
    preparation_time: float
    metadata: Dict[str, Any]

class FeatureService:
    """
    Feature service that wraps FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer.

    Responsibilities:
    - Wrap FeaturePreprocessor, FeatureSelector, and FeatureAnalyzer
    - Handle scaling (RobustScaler), PCA/UMAP embedding
    - Expose API: prepare_features(data) → clean feature matrix ready for clustering
    """

    def __init__(self, verbose: bool = True):
        """Initialize the feature service."""
        self.verbose = verbose
        self.logger = get_logger('FeatureService')

        # Feature preparation components
        self.scaler = None
        self.pca = None
        self.umap_reducer = None

        # Hardware service integration
        try:
            from .hardware_service import HardwareService
            self.hardware_service = HardwareService(verbose=self.verbose)
            self.hardware_integration_enabled = True
        except ImportError:
            self.hardware_service = None
            self.hardware_integration_enabled = False

        # Mac M1 Hardware Optimizations
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=2.0)  # Conservative limit for feature processing
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.cpu_optimizer.set_conservative_mode()  # Use conservative mode for feature processing
                tprint("🧠 Mac M1 hardware optimizations initialized for feature service", "INFO")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize hardware optimizations: {e}", "WARNING")

        # Feature tracking
        self.feature_history = []
        self.performance_metrics = {
            "total_preparation_time": 0.0,
            "scaling_time": 0.0,
            "embedding_time": 0.0,
            "feature_reduction_rate": 0.0,
            "hardware_accelerations": 0,
            "memory_optimizations": 0
        }

    async def prepare_features(
        self,
        market_data: pd.DataFrame,
        config: Any = None
    ) -> FeaturePreparationResult:
        """
        Prepare features for clustering.

        Args:
            market_data: Market data for feature extraction
            config: Configuration parameters

        Returns:
            FeaturePreparationResult with clean feature matrix
        """
        try:
            start_time = time.time()
            tprint("🔧 Starting feature preparation", "INFO")

            # Start memory monitoring for feature preparation
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.start_monitoring()
                    tprint("🧠 Memory monitoring started for feature preparation", "INFO")
                except Exception as e:
                    tprint(f"⚠️ Memory monitoring failed: {e}", "WARNING")

            # Optimize market data for memory efficiency
            if self.memory_optimizer and hasattr(market_data, 'memory_usage'):
                try:
                    market_data = self.memory_optimizer.optimize_dataframe_memory(market_data)
                    tprint("🧠 Market data memory optimized for feature preparation", "INFO")
                except Exception as e:
                    tprint(f"⚠️ Data optimization failed: {e}", "WARNING")

            # Step 1: Extract features using shared utilities
            feature_config = self._create_feature_config(config)

            # Validate market data before feature preparation
            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data is None or empty in feature preparation")

            shared_result = await self._prepare_features_shared(market_data, feature_config)

            # Validate shared result
            if shared_result is None or not hasattr(shared_result, 'features') or shared_result.features is None:
                raise ValueError("Shared feature preparation returned None or invalid result")

            if shared_result.features.size == 0:
                raise ValueError("Shared feature preparation returned empty features array")

            # Step 2: Apply scaling and normalization
            scaled_features, scaling_time = await self._apply_scaling(shared_result.features)

            # Step 3: Apply dimensionality reduction (PCA/UMAP)
            final_features, embedding_time = await self._apply_embedding(
                scaled_features, shared_result.feature_names, config
            )

            # Step 4: Validate final features
            validation_results = self._validate_features(final_features, market_data)

            # Record performance metrics
            total_time = time.time() - start_time
            self.performance_metrics["total_preparation_time"] = total_time
            self.performance_metrics["scaling_time"] = scaling_time
            self.performance_metrics["embedding_time"] = embedding_time

            # Calculate feature reduction rate
            original_count = shared_result.features.shape[1]
            final_count = final_features.shape[1]
            reduction_rate = (original_count - final_count) / original_count
            self.performance_metrics["feature_reduction_rate"] = reduction_rate

            # Log comprehensive feature preparation summary
            tprint(f"📊 Feature Preparation Summary:", "INFO")
            tprint(f"  📈 Original features: {original_count} → Final features: {final_count}", "INFO")
            tprint(f"  📉 Feature reduction: {reduction_rate*100:.1f}%", "INFO")
            tprint(f"  🎯 Embedding method: {self._get_embedding_method()}", "INFO")
            tprint(f"  ✅ Validation: {'PASSED' if validation_results.get('valid', False) else 'FAILED'}", "INFO")

            # Create result with proper column names for reduced features
            if final_features.shape[1] == len(shared_result.feature_names):
                # No dimensionality reduction, use original feature names
                feature_names = shared_result.feature_names
            else:
                # Dimensionality reduction applied, create meaningful names based on method
                if self._get_embedding_method() == "PCA":
                    # Create PCA component names with variance information
                    if hasattr(self, 'pca') and self.pca is not None:
                        feature_names = [f"PC{i+1}_var{self.pca.explained_variance_ratio_[i]:.3f}"
                                       for i in range(final_features.shape[1])]
                    else:
                        feature_names = [f"PC{i+1}" for i in range(final_features.shape[1])]
                elif self._get_embedding_method() == "UMAP":
                    feature_names = [f"UMAP_dim{i+1}" for i in range(final_features.shape[1])]
                else:
                    feature_names = [f"embedding_{i+1}" for i in range(final_features.shape[1])]

            result = FeaturePreparationResult(
                features=final_features,
                feature_names=feature_names,
                feature_scores={},
                dropped_features=[],
                preparation_time=total_time,
                metadata={
                    "original_feature_count": original_count,
                    "final_feature_count": final_count,
                    "feature_reduction_rate": self.performance_metrics["feature_reduction_rate"],
                    "scaling_time": scaling_time,
                    "scaling_method": "robust",
                    "embedding_time": embedding_time,
                    "embedding_method": "pca",
                    "validation_passed": validation_results.get("passed", True),
                    "validation_results": validation_results,
                    "performance_metrics": self.performance_metrics
                }
            )

            # Track feature history
            self._track_feature_preparation(result)

            # Final memory cleanup for feature preparation
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.force_garbage_collection()
                    tprint("🧠 Final memory cleanup completed for feature preparation", "INFO")
                except Exception as e:
                    tprint(f"⚠️ Final cleanup failed: {e}", "WARNING")

            # Stop memory monitoring
            if self.memory_optimizer:
                try:
                    self.memory_optimizer.stop_monitoring()
                    tprint("🧠 Memory monitoring stopped for feature preparation", "INFO")
                except Exception as e:
                    tprint(f"⚠️ Memory monitoring stop failed: {e}", "WARNING")

            tprint(f"✅ Feature preparation completed in {total_time:.2f}s", "SUCCESS")
            tprint(f"📊 Features: {original_count} → {final_count} (reduction: {result.metadata['performance_metrics']['feature_reduction_rate']:.1%})", "INFO")

            return result

        except Exception as e:
            tprint(f"❌ Feature preparation failed: {e}", "ERROR")
            raise ValueError(f"Feature preparation failed: {e}")

    def _create_feature_config(self, config: Any) -> FeatureConfig:
        """Create feature configuration from provided config."""
        return FeatureConfig(
            feature_categories=getattr(config, 'feature_categories', [
                'regime_volatility',
                'regime_volume',
                'regime_structural_trend',
                'regime_statistical'
            ]),
            use_standardized_features=getattr(config, 'use_standardized_features', True),
            drop_highly_correlated=getattr(config, 'drop_highly_correlated', True),
            correlation_threshold=getattr(config, 'correlation_threshold', 0.95)
        )

    async def _prepare_features_shared(
        self,
        market_data: pd.DataFrame,
        feature_config: FeatureConfig
    ):
        """Prepare features using shared utilities."""
        try:
            tprint("📊 Preparing features using shared utilities", "INFO")

            # Validate inputs
            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data is None or empty in shared feature preparation")

            if feature_config is None:
                raise ValueError("Feature config is None in shared feature preparation")

            # Use shared feature preparation
            result = prepare_market_features(
                market_data=market_data,
                feature_config=feature_config,
                return_metadata=True
            )

            # Validate result
            if result is None:
                raise ValueError("Shared feature preparation returned None")

            # Handle both return types: FeaturePreparationResult or numpy array
            if hasattr(result, 'features_array'):
                # It's a FeaturePreparationResult from shared_utils
                features = result.features_array
                feature_names = list(result.features_df.columns) if hasattr(result, 'features_df') and result.features_df is not None else []
                feature_scores = {}
                dropped_features = []
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                preparation_time = 0.0
            elif hasattr(result, 'features'):
                # It's a FeaturePreparationResult from feature_service
                features = result.features
                feature_names = result.feature_names
                feature_scores = result.feature_scores if hasattr(result, 'feature_scores') else {}
                dropped_features = result.dropped_features if hasattr(result, 'dropped_features') else []
                metadata = result.metadata if hasattr(result, 'metadata') else {}
                preparation_time = result.preparation_time if hasattr(result, 'preparation_time') else 0.0
            else:
                # It's a numpy array
                features = result
                feature_names = []
                feature_scores = {}
                dropped_features = []
                metadata = {}
                preparation_time = 0.0

            tprint(f"✅ Shared utilities prepared {features.shape[1]} features", "SUCCESS")

            # Create a proper FeaturePreparationResult object for consistency
            return FeaturePreparationResult(
                features=features,
                feature_names=feature_names,
                feature_scores=feature_scores,
                dropped_features=dropped_features,
                preparation_time=preparation_time,
                metadata={
                    **metadata,
                    "scaling_method": "robust",  # Default scaling method for shared utilities
                    "embedding_method": "none"   # No embedding applied in shared utilities
                }
            )

        except Exception as e:
            tprint(f"❌ Shared feature preparation failed: {e}", "ERROR")
            raise

    async def _apply_scaling(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply scaling to features with hardware optimization."""
        try:
            start_time = time.time()
            tprint("⚖️ Applying feature scaling", "INFO")

            # Apply memory optimization if hardware service is available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    features, optimization_info = self.hardware_service.optimize_memory(features)
                    if optimization_info.get("hardware_optimization_used", False):
                        self.performance_metrics["memory_optimizations"] += 1
                        tprint(f"🧠 Memory optimization applied during scaling", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Memory optimization failed during scaling: {e}", "WARNING")

            # Import and initialize scaler
            from sklearn.preprocessing import RobustScaler

            # Use RobustScaler for financial data (handles outliers well)
            self.scaler = RobustScaler()

            # Fit and transform
            scaled_features = self.scaler.fit_transform(features)

            scaling_time = time.time() - start_time

            tprint(f"✅ Scaling completed in {scaling_time:.3f}s", "SUCCESS")
            return scaled_features, scaling_time

        except Exception as e:
            tprint(f"❌ Feature scaling failed: {e}", "ERROR")
            raise

    async def _apply_embedding(
        self,
        features: np.ndarray,
        feature_names: List[str],
        config: Any
    ) -> Tuple[np.ndarray, float]:
        """Apply dimensionality reduction (PCA/UMAP)."""
        try:
            start_time = time.time()
            tprint("🗺️ Applying dimensionality reduction", "INFO")

            # Check memory pressure before dimensionality reduction
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                memory_optimizer = get_m1_memory_optimizer()
                memory_pressure = getattr(memory_optimizer, 'memory_pressure', 0.0)

                if memory_pressure > 0.85:  # Very high memory pressure threshold
                    tprint(f"🧠 Very high memory pressure detected ({memory_pressure:.2f}), skipping dimensionality reduction", "WARNING")
                    return features, 0.0
            except Exception as e:
                tprint(f"Could not check memory pressure: {e}, proceeding with dimensionality reduction", "WARNING")

            # Check if dimensionality reduction is needed
            n_features = features.shape[1]
            n_samples = features.shape[0]
            target_features = getattr(config, 'target_features', min(20, n_features - 1))

            # Log embedding configuration
            tprint(f"  📊 Input: {n_samples} samples × {n_features} features", "DEBUG")
            tprint(f"  🎯 Target: {target_features} components", "DEBUG")
            tprint(f"  📉 Max reduction: {((n_features - target_features) / n_features * 100):.1f}%", "DEBUG")

            if n_features <= target_features:
                tprint(f"📊 No reduction needed: {n_features} features", "INFO")
                return features, 0.0

            # Try UMAP first (better for non-linear relationships)
            tprint(f"🔍 Attempting UMAP reduction...", "DEBUG")
            umap_features = await self._try_umap_reduction(features, target_features)

            if umap_features is not None:
                embedding_time = time.time() - start_time
                tprint(f"✅ UMAP reduction: {n_features} → {umap_features.shape[1]} features", "SUCCESS")
                tprint(f"  📊 Dimensionality reduction: {((n_features - umap_features.shape[1]) / n_features * 100):.1f}%", "INFO")
                return umap_features, embedding_time

            # Fallback to PCA
            tprint(f"🔄 UMAP not available, using PCA fallback", "INFO")
            pca_features = await self._apply_pca_reduction(features, target_features, feature_names)

            embedding_time = time.time() - start_time
            tprint(f"✅ PCA reduction: {n_features} → {pca_features.shape[1]} features", "SUCCESS")
            tprint(f"  📊 Dimensionality reduction: {((n_features - pca_features.shape[1]) / n_features * 100):.1f}%", "INFO")
            return pca_features, embedding_time

        except Exception as e:
            tprint(f"❌ Dimensionality reduction failed: {e}", "ERROR")
            tprint("⚠️ Returning original features", "WARNING")
            return features, 0.0

    async def _try_umap_reduction(self, features: np.ndarray, target_features: int) -> Optional[np.ndarray]:
        """Try UMAP reduction as primary method with hardware acceleration."""
        try:
            import umap  # type: ignore

            if not hasattr(umap, 'UMAP'):
                return None

            # Apply hardware acceleration if available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    # Try to use GPU acceleration for UMAP
                    neighbors_result, acceleration_info = self.hardware_service.accelerate_neighbors(
                        features, n_neighbors=min(15, features.shape[0] // 10)
                    )

                    if acceleration_info.get("hardware_acceleration_used", False):
                        self.performance_metrics["hardware_accelerations"] += 1
                        tprint(f"🏎️ Hardware acceleration used for UMAP neighbors computation", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware acceleration failed for UMAP: {e}", "WARNING")

            # Initialize UMAP reducer
            self.umap_reducer = umap.UMAP(
                n_components=target_features,
                random_state=42,
                n_neighbors=min(15, features.shape[0] // 10),
                min_dist=0.1,
                metric='euclidean'
            )

            # Fit and transform
            reduced_features = self.umap_reducer.fit_transform(features)

            return reduced_features

        except ImportError:
            tprint("📦 UMAP not available, using PCA fallback", "INFO")
            return None
        except Exception as e:
            tprint(f"⚠️ UMAP reduction failed: {e}, using PCA fallback", "WARNING")
            return None

    async def _apply_pca_reduction(self, features: np.ndarray, target_features: int, feature_names: List[str] = None) -> np.ndarray:
        """Apply PCA reduction as fallback method."""
        try:
            from sklearn.decomposition import PCA

            # Check memory pressure before PCA fitting
            try:
                from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
                memory_optimizer = get_m1_memory_optimizer()
                memory_pressure = getattr(memory_optimizer, 'memory_pressure', 0.0)

                if memory_pressure > 0.8:  # High memory pressure threshold
                    tprint(f"🧠 High memory pressure detected ({memory_pressure:.2f}), using simplified PCA", "WARNING")
                    # Use fewer components to reduce memory usage
                    target_features = min(target_features, 5)
                    tprint(f"📉 Reduced target components to {target_features} due to memory pressure", "INFO")
            except Exception as e:
                tprint(f"Could not check memory pressure: {e}, proceeding with normal PCA", "WARNING")

            # Log PCA initialization details
            n_samples, n_features = features.shape
            tprint(f"🔧 Initializing PCA reduction", "INFO")
            tprint(f"  📊 Input features: {n_features} dimensions, {n_samples} samples", "DEBUG")
            tprint(f"  🎯 Target components: {target_features}", "DEBUG")
            tprint(f"  🔄 Random state: 42 (for reproducibility)", "DEBUG")

            # Initialize PCA
            self.pca = PCA(n_components=target_features, random_state=42)

            # Log PCA fitting process
            tprint(f"🔍 Fitting PCA to data...", "INFO")

            # Fit and transform
            reduced_features = self.pca.fit_transform(features)

            tprint(f"  ✅ PCA fitting completed", "DEBUG")

            # Log PCA results and analysis
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            total_variance_explained = cumulative_variance[-1]

            tprint(f"📈 PCA Reduction Results:", "INFO")
            tprint(f"  📊 Original features: {n_features} → Reduced features: {reduced_features.shape[1]}", "INFO")
            tprint(f"  📉 Total variance explained: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)", "INFO")
            tprint(f"  📊 Feature reduction: {((n_features - reduced_features.shape[1]) / n_features * 100):.1f}%", "INFO")

            # Log component-wise variance explained with feature contributions
            tprint(f"🔍 Component Analysis:", "DEBUG")
            for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
                feature_name = f"PC{i+1}_var{var_ratio:.3f}"
                tprint(f"  {feature_name}: {var_ratio:.4f} ({var_ratio*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)", "DEBUG")

                # Analyze which original features contribute most to this component
                if hasattr(self.pca, 'components_') and feature_names and len(feature_names) > 0:
                    component_loadings = self.pca.components_[i]
                    # Get top contributing features (absolute values)
                    top_features_idx = np.argsort(np.abs(component_loadings))[-5:][::-1]  # Top 5

                    tprint(f"    🎯 Top contributing features:", "DEBUG")
                    for j, feat_idx in enumerate(top_features_idx):
                        if feat_idx < len(feature_names):
                            feat_name = feature_names[feat_idx]
                            loading = component_loadings[feat_idx]
                            # Categorize feature type
                            feat_type = self._categorize_feature(feat_name)
                            tprint(f"      {j+1}. {feat_name} ({feat_type}): {loading:.4f}", "DEBUG")

            # Log top components that explain most variance
            top_components = np.argsort(explained_variance_ratio)[::-1][:5]
            tprint(f"🏆 Top 5 Components by Variance:", "DEBUG")
            for i, comp_idx in enumerate(top_components):
                feature_name = f"PC{comp_idx+1}_var{explained_variance_ratio[comp_idx]:.3f}"
                tprint(f"  {i+1}. {feature_name}: {explained_variance_ratio[comp_idx]:.4f} ({explained_variance_ratio[comp_idx]*100:.2f}%)", "DEBUG")

            # Analyze feature type composition for top components
            tprint(f"📊 PCA Component Feature Analysis:", "INFO")
            for i in range(min(3, reduced_features.shape[1])):  # Analyze top 3 components
                if hasattr(self.pca, 'components_') and feature_names and len(feature_names) > 0:
                    component_loadings = self.pca.components_[i]
                    # Get all contributing features (not just top 5)
                    feature_contributions = []
                    for j, loading in enumerate(component_loadings):
                        if j < len(feature_names):
                            feat_name = feature_names[j]
                            feat_type = self._categorize_feature(feat_name)
                            feature_contributions.append((feat_name, feat_type, abs(loading)))

                    # Sort by contribution strength
                    feature_contributions.sort(key=lambda x: x[2], reverse=True)

                    # Count feature types
                    type_counts = {}
                    for _, feat_type, _ in feature_contributions:
                        type_counts[feat_type] = type_counts.get(feat_type, 0) + 1

                    # Show composition
                    component_name = f"PC{i+1}_var{explained_variance_ratio[i]:.3f}"
                    tprint(f"  🎯 {component_name} composition:", "INFO")
                    for feat_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(feature_contributions)) * 100
                        tprint(f"    {feat_type}: {count} features ({percentage:.1f}%)", "INFO")

            # Log data quality metrics
            reduced_mean = np.mean(reduced_features, axis=0)
            reduced_std = np.std(reduced_features, axis=0)
            tprint(f"📊 Reduced Feature Statistics:", "DEBUG")
            tprint(f"  Mean range: [{np.min(reduced_mean):.4f}, {np.max(reduced_mean):.4f}]", "DEBUG")
            tprint(f"  Std range: [{np.min(reduced_std):.4f}, {np.max(reduced_std):.4f}]", "DEBUG")

            # Check for potential issues
            if total_variance_explained < 0.8:
                tprint(f"⚠️ Low variance explained ({total_variance_explained:.2f}%) - consider more components", "WARNING")

            if np.any(np.isnan(reduced_features)):
                tprint(f"❌ NaN values detected in reduced features!", "ERROR")

            if np.any(np.isinf(reduced_features)):
                tprint(f"❌ Infinite values detected in reduced features!", "ERROR")

            return reduced_features

        except Exception as e:
            tprint(f"❌ PCA reduction failed: {e}", "ERROR")
            raise

    def _validate_features(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate prepared features."""
        try:
            tprint("🔍 Validating prepared features", "INFO")

            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            # Check basic properties with validation
            if not validate_finite(features.shape[0], "feature_count"):
                validation_results["valid"] = False
                validation_results["issues"].append("Invalid feature count")
            elif features.shape[0] == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("No samples in features")

            if not validate_finite(features.shape[1], "feature_dimensions"):
                validation_results["valid"] = False
                validation_results["issues"].append("Invalid feature dimensions")
            elif features.shape[1] == 0:
                validation_results["valid"] = False
                validation_results["issues"].append("No features available")

            if features.shape[0] < 10:
                validation_results["warnings"].append("Very few samples for clustering")

            if features.shape[1] < 2:
                validation_results["valid"] = False
                validation_results["issues"].append("Insufficient features for clustering")

            # Check for NaN values with safe operations
            nan_count = int(np.sum(np.isnan(features)))
            if nan_count > 0:
                validation_results["warnings"].append(f"Features contain {nan_count} NaN values")

            # Check for infinite values with safe operations
            inf_count = int(np.sum(np.isinf(features)))
            if inf_count > 0:
                validation_results["warnings"].append(f"Features contain {inf_count} infinite values")

            # Check feature variance (avoid constant features) with safe math
            try:
                feature_variances = np.var(features, axis=0)
                constant_features = int(np.sum(feature_variances < 1e-8))
                if constant_features > 0:
                    validation_results["warnings"].append(f"{constant_features} constant features detected")
            except Exception as e:
                validation_results["warnings"].append(f"Could not calculate feature variances: {e}")

            tprint(f"✅ Feature validation completed: {len(validation_results['issues'])} issues, {len(validation_results['warnings'])} warnings", "SUCCESS")
            return validation_results

        except Exception as e:
            tprint(f"❌ Feature validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature by its name to identify type (volatility, momentum, trend, etc.)."""
        feature_name_lower = feature_name.lower()

        # Volatility indicators
        if any(term in feature_name_lower for term in ['vol', 'volatility', 'atr', 'std', 'dev', 'range', 'bb', 'bollinger']):
            return "VOLATILITY"

        # Momentum indicators
        elif any(term in feature_name_lower for term in ['rsi', 'momentum', 'roc', 'rate_of_change', 'stoch', 'stochastic', 'williams', 'cci']):
            return "MOMENTUM"

        # Trend indicators
        elif any(term in feature_name_lower for term in ['ma', 'moving_average', 'ema', 'sma', 'trend', 'macd', 'adx', 'dmi', 'aroon']):
            return "TREND"

        # Volume indicators
        elif any(term in feature_name_lower for term in ['volume', 'vol', 'obv', 'ad', 'accumulation', 'distribution', 'mfi', 'money_flow']):
            return "VOLUME"

        # Price-based features
        elif any(term in feature_name_lower for term in ['price', 'close', 'open', 'high', 'low', 'return', 'change', 'pct']):
            return "PRICE"

        # Statistical features
        elif any(term in feature_name_lower for term in ['skew', 'kurt', 'stat', 'corr', 'correlation', 'beta', 'alpha']):
            return "STATISTICAL"

        # Regime features
        elif any(term in feature_name_lower for term in ['regime', 'state', 'phase', 'cycle']):
            return "REGIME"

        # Technical patterns
        elif any(term in feature_name_lower for term in ['pattern', 'signal', 'crossover', 'breakout', 'support', 'resistance']):
            return "PATTERN"

        # Default category
        else:
            return "OTHER"

    def _get_embedding_method(self) -> str:
        """Get the current embedding method name."""
        if self.umap_reducer is not None:
            return "UMAP"
        elif self.pca is not None:
            return "PCA"
        else:
            return "None"

    def _track_feature_preparation(self, result: FeaturePreparationResult):
        """Track feature preparation for analysis."""
        try:
            self.feature_history.append({
                "timestamp": time.time(),
                "original_features": result.metadata["original_feature_count"],
                "final_features": result.metadata["final_feature_count"],
                "preparation_time": result.preparation_time,
                "scaling_method": result.metadata["scaling_method"],
                "embedding_method": result.metadata["embedding_method"],
                "validation_issues": len(result.metadata["validation_results"]["issues"]),
                "validation_warnings": len(result.metadata["validation_results"]["warnings"])
            })

            # Keep only last 10 entries
            if len(self.feature_history) > 10:
                self.feature_history = self.feature_history[-10:]

        except Exception as e:
            tprint(f"⚠️ Feature tracking failed: {e}", "WARNING")

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature preparation statistics."""
        if not self.feature_history:
            return {"message": "No feature preparation history available"}

        # Calculate statistics across all preparations
        prep_times = [h["preparation_time"] for h in self.feature_history]
        feature_counts = [h["final_features"] for h in self.feature_history]

        return {
            "total_preparations": len(self.feature_history),
            "average_preparation_time": np.mean(prep_times),
            "min_preparation_time": np.min(prep_times),
            "max_preparation_time": np.max(prep_times),
            "average_feature_count": np.mean(feature_counts),
            "min_feature_count": np.min(feature_counts),
            "max_feature_count": np.max(feature_counts),
            "performance_metrics": self.performance_metrics,
            "recent_history": self.feature_history[-3:]  # Last 3 preparations
        }

    def clear_feature_cache(self):
        """Clear feature preparation cache and reset state."""
        try:
            self.scaler = None
            self.pca = None
            self.umap_reducer = None
            self.feature_history.clear()

            tprint("🧹 Feature cache cleared", "INFO")

        except Exception as e:
            tprint(f"⚠️ Cache clearing failed: {e}", "WARNING")

    async def prepare_features_for_clustering(
        self,
        market_data: pd.DataFrame,
        clustering_config: Any = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepare features specifically for clustering.

        Args:
            market_data: Market data for feature extraction
            clustering_config: Clustering-specific configuration

        Returns:
            Tuple of (features, feature_names, metadata)
        """
        try:
            # Use clustering-specific configuration if provided
            if clustering_config:
                config = clustering_config
            else:
                # Create default clustering configuration
                config = type('Config', (), {
                    'feature_categories': ['regime_volatility', 'regime_volume', 'regime_structural_trend', 'regime_statistical'],
                    'use_standardized_features': True,
                    'drop_highly_correlated': True,
                    'correlation_threshold': 0.95,
                    'target_features': 20
                })()

            # Prepare features
            result = await self.prepare_features(market_data, config)

            return (
                result.features,
                result.feature_names,
                result.metadata
            )

        except Exception as e:
            tprint(f"❌ Clustering feature preparation failed: {e}", "ERROR")
            raise
