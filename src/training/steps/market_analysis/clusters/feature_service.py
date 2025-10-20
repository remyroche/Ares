from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)

"""
Feature Service for NAS-TAS Clustering.

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
    (features)
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
                
                memory_optimizer = get_integrated_hardware_manager()
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
                
                memory_optimizer = get_integrated_hardware_manager()
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
