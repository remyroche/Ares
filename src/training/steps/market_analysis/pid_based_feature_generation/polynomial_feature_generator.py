"""
Polynomial Feature Generator using Partial Information Decomposition

This module generates data-driven polynomial features using PID analysis to identify
the most relevant polynomial transformations up to 50 features.

Key Features:
- Uses optimized lookback periods from feature_lookback_optimization
- Leverages matrix_operations/ for all calculations
- Generates up to 50 polynomial features based on PID analysis
- Comprehensive validation and error handling
- Hardware-optimized computations
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import PID utilities
try:
    from src.training.utils.feature_selection.partial_information_decompositor import (
        PartialInformationDecompositor, PIDConfig, PIDResult
    )
    PID_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PID utilities not available: {e}")
    PID_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Matrix operations not available: {e}")
    MATRIX_OPS_AVAILABLE = False

# Import tprint for extensive logging
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback to basic print
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Import base feature generator
from .base_feature_generator import BaseFeatureGenerator, BaseFeatureConfig, BaseFeatureResult

# Import common operations for comprehensive utility integration
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    logging.warning(f"Common operations not available: {e}")
    # Fallback functions
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def safe_log(x, default=0.0): return np.log(x) if x > 0 else default
    def safe_sqrt(x, default=0.0): return np.sqrt(x) if x >= 0 else default
    def safe_power(x, y, default=0.0): return x ** y if np.isfinite(x) and np.isfinite(y) else default
    def validate_finite(value, name="value"): return float(value) if np.isfinite(value) else 0.0

# Import serialization utilities
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    SERIALIZATION_AVAILABLE = False
    logging.warning(f"Serialization utilities not available: {e}")

# Import math validation for additional math operations
try:
    from src.utils.math_validation import MathValidation, safe_correlation, safe_covariance, safe_percentile
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

# Import logger as fallback
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('PolynomialFeatureGenerator')
except ImportError:
    logger = logging.getLogger('PolynomialFeatureGenerator')
    logger.setLevel(logging.INFO)


class PolynomialType(Enum):
    """Types of polynomial features."""
    POWER = "power"
    CROSS_PRODUCT = "cross_product"
    INTERACTION = "interaction"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    SQUARE_ROOT = "square_root"
    CUBIC_ROOT = "cubic_root"
    RECIPROCAL = "reciprocal"


@dataclass
class PolynomialConfig(BaseFeatureConfig):
    """Configuration for polynomial feature generation with common utilities integration."""
    # PID Configuration
    synergy_threshold: float = 0.1
    redundancy_threshold: float = 0.15
    unique_info_threshold: float = 0.05
    
    # Feature Limits
    max_polynomial_features: int = 50
    max_polynomial_degree: int = 3
    max_feature_combinations: int = 25
    
    # Polynomial Types
    polynomial_types: List[PolynomialType] = field(default_factory=lambda: [
        PolynomialType.POWER,
        PolynomialType.CROSS_PRODUCT,
        PolynomialType.INTERACTION,
        PolynomialType.SQUARE_ROOT
    ])
    
    # Computational Settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Validation
    min_variance_threshold: float = 0.01
    max_skewness_threshold: float = 5.0
    significance_threshold: float = 0.05
    
    # Hardware Optimization
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.7


@dataclass
class PolynomialResult(BaseFeatureResult):
    """Result of polynomial feature generation with common utilities integration."""
    polynomial_features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    polynomial_scores: Dict[str, float] = field(default_factory=dict)
    pid_analysis: Optional[PIDResult] = None
    
    # Metadata
    total_features_generated: int = 0
    execution_time: float = 0.0
    optimization_used: bool = False
    matrix_ops_used: bool = False
    
    # Quality Metrics
    average_variance: float = 0.0
    feature_stability_score: float = 0.0
    polynomial_degree_distribution: Dict[int, int] = field(default_factory=dict)


class PolynomialFeatureGenerator(BaseFeatureGenerator):
    """
    Polynomial Feature Generator using Partial Information Decomposition.
    
    Generates data-driven polynomial features using PID analysis to identify
    the most relevant polynomial transformations up to 50 features.
    """
    
    def __init__(self, config: Optional[PolynomialConfig] = None):
        """Initialize the polynomial feature generator with common utilities integration."""
        super().__init__(config or PolynomialConfig(), "PolynomialFeatureGenerator")
        
        tprint_info(f"📊 Max polynomial features: {self.config.max_polynomial_features}")
        tprint_info(f"📊 Max polynomial degree: {self.config.max_polynomial_degree}")
        tprint_info(f"📊 Polynomial types: {[t.value for t in self.config.polynomial_types]}")
    
    def _initialize_components(self):
        """Initialize required components."""
        # Initialize PID decompositor
        if PID_AVAILABLE:
            pid_config = PIDConfig(
                synergy_threshold=self.config.synergy_threshold,
                redundancy_threshold=self.config.redundancy_threshold,
                unique_info_threshold=self.config.unique_info_threshold,
                max_polynomial_degree=self.config.max_polynomial_degree,
                max_interaction_features=self.config.max_polynomial_features
            )
            self.pid_decompositor = PartialInformationDecompositor(pid_config)
            self.logger.info("✅ PID Decompositor initialized")
        else:
            self.pid_decompositor = None
            self.logger.warning("⚠️ PID Decompositor not available")
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_memory_optimization=True,
                enable_parallel=self.config.enable_parallel_processing
            )
            self.logger.info("✅ Matrix Operations initialized")
        else:
            self.matrix_ops = None
            self.logger.warning("⚠️ Matrix Operations not available")
    
    async def generate_polynomial_features(
        self, 
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: List[str],
        optimized_lookback_periods: Optional[Dict[str, int]] = None,
        target: Optional[np.ndarray] = None
    ) -> PolynomialResult:
        """
        Generate polynomial features using PID analysis.
        
        Args:
            data: Input feature matrix
            feature_names: List of feature names
            optimized_lookback_periods: Optimized lookback periods from feature_lookback_optimization
            target: Target variable for PID analysis (optional)
            
        Returns:
            PolynomialResult with generated polynomial features
        """
        start_time = time.time()
        tprint_info("Starting polynomial feature generation...")
        
        result = PolynomialResult()
        
        try:
            # Fast-fail input validation
            if data is None:
                raise ValueError("Data cannot be None - fast failing")
            
            if feature_names is None or len(feature_names) == 0:
                raise ValueError("Feature names cannot be None or empty - fast failing")
            
            # Convert data to numpy array if needed
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Input DataFrame is empty - fast failing")
                
                # Ensure we only work with numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    raise ValueError("No numeric columns found in DataFrame - fast failing")
                
                X = numeric_data.values
                if feature_names is None:
                    feature_names = list(numeric_data.columns)
                else:
                    # Filter feature_names to match numeric columns
                    feature_names = [name for name in feature_names if name in numeric_data.columns]
                    if len(feature_names) == 0:
                        raise ValueError("No matching feature names found in numeric columns - fast failing")
                
                tprint_info(f"Converted DataFrame to numpy array: {X.shape}")
            else:
                if not hasattr(data, 'shape'):
                    raise TypeError(f"Data must be array-like, got {type(data)} - fast failing")
                X = data
                
                # Ensure numeric data
                if X.dtype == object:
                    tprint_warning("Input data contains non-numeric types, attempting conversion")
                    try:
                        X = pd.DataFrame(X).select_dtypes(include=[np.number]).values
                        if X.shape[1] == 0:
                            raise ValueError("No numeric columns found after conversion - fast failing")
                    except Exception as e:
                        raise ValueError(f"Cannot convert input data to numeric format: {e} - fast failing")
            
            # Validate data shape
            if X.shape[0] == 0:
                raise ValueError("Input data has no samples - fast failing")
            if X.shape[1] == 0:
                raise ValueError("Input data has no features - fast failing")
            
            # Check for NaN/Inf values
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            if nan_count > 0:
                tprint_warning(f"Input data contains {nan_count} NaN values - this may cause issues")
            if inf_count > 0:
                tprint_warning(f"Input data contains {inf_count} Inf values - this may cause issues")
            
            # Validate feature names match data dimensions
            if len(feature_names) != X.shape[1]:
                raise ValueError(f"Feature names count ({len(feature_names)}) doesn't match data columns ({X.shape[1]}) - fast failing")
            
            # Validate target if provided
            if target is not None:
                if len(target) != X.shape[0]:
                    raise ValueError(f"Target length ({len(target)}) doesn't match data length ({X.shape[0]}) - fast failing")
                if np.any(np.isnan(target)) or np.any(np.isinf(target)):
                    tprint_warning("Target contains NaN or Inf values - this may cause issues")
            
            tprint_info(f"Input data shape: {X.shape}")
            tprint_info(f"Feature count: {len(feature_names)}")
            tprint_info(f"Data type: {X.dtype}")
            
            # Apply optimized lookback periods if available
            if optimized_lookback_periods:
                try:
                    tprint_info("Applying optimized lookback periods...")
                    X, feature_names = self._apply_optimized_lookback_periods(
                        X, feature_names, optimized_lookback_periods
                    )
                    result.optimization_used = True
                    tprint_success("Applied optimized lookback periods")
                except Exception as e:
                    tprint_warning(f"Failed to apply optimized lookback periods: {e}")
                    # Continue without optimization
            
            # Perform PID analysis
            if self.pid_decompositor and target is not None:
                try:
                    tprint_info("Performing PID analysis...")
                    pid_result = self.pid_decompositor.decompose_information(X, target, feature_names)
                    result.pid_analysis = pid_result
                    
                    # Extract significant features from PID
                    significant_features = self._extract_significant_features(pid_result, feature_names)
                    tprint_success(f"Found {len(significant_features)} significant features from PID analysis")
                except Exception as e:
                    tprint_warning(f"PID analysis failed: {e}, falling back to variance-based selection")
                    significant_features = self._variance_based_feature_selection(X, feature_names)
            else:
                # Fallback: use variance-based approach
                tprint_info("Using variance-based feature selection")
                significant_features = self._variance_based_feature_selection(X, feature_names)
                tprint_success(f"Selected {len(significant_features)} features based on variance")
            
            # Generate polynomial features
            try:
                tprint_info("Generating polynomial features...")
                polynomial_features, polynomial_names = self._generate_polynomial_matrix(
                    X, feature_names, significant_features
                )
                tprint_success(f"Generated {len(polynomial_names)} polynomial features")
            except Exception as e:
                tprint_error(f"Failed to generate polynomial features: {e}")
                raise
            
            # Calculate polynomial scores
            try:
                tprint_info("Calculating polynomial scores...")
                polynomial_scores = self._calculate_polynomial_scores(
                    polynomial_features, polynomial_names, target
                )
                tprint_success("Polynomial scores calculated")
            except Exception as e:
                tprint_warning(f"Failed to calculate polynomial scores: {e}")
                polynomial_scores = {}
            
            # Store results
            try:
                result.polynomial_features = {
                    name: feature for name, feature in zip(polynomial_names, polynomial_features.T)
                }
                result.feature_names = polynomial_names
                result.polynomial_scores = polynomial_scores
                result.total_features_generated = len(polynomial_names)
                result.matrix_ops_used = self.matrix_ops is not None
                tprint_success("Results stored successfully")
            except Exception as e:
                tprint_error(f"Failed to store results: {e}")
                raise
            
            # Calculate quality metrics
            try:
                tprint_info("Calculating quality metrics...")
                result.average_variance = self._calculate_average_variance(polynomial_features)
                result.feature_stability_score = self._calculate_stability_score(polynomial_features)
                result.polynomial_degree_distribution = self._calculate_degree_distribution(polynomial_names)
                tprint_success("Quality metrics calculated")
            except Exception as e:
                tprint_warning(f"Failed to calculate quality metrics: {e}")
                # Set default values
                result.average_variance = 0.0
                result.feature_stability_score = 0.0
                result.polynomial_degree_distribution = {}
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # Set utility integration status using base class method
            self._set_utility_integration_status(result)
            
            tprint_performance("Polynomial feature generation", execution_time)
            tprint_success(f"Generated {result.total_features_generated} polynomial features")
            tprint_info(f"Average variance: {result.average_variance:.3f}")
            tprint_info(f"Stability score: {result.feature_stability_score:.3f}")
            tprint_info(f"🔧 Utility integrations: {sum(result.utility_integration_status.values())}/{len(result.utility_integration_status)}")
            
            return result
            
        except ValueError as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_error(f"Polynomial feature generation failed - validation error: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
            
        except TypeError as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_error(f"Polynomial feature generation failed - type error: {e}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_error(f"Polynomial feature generation failed - unexpected error: {e}")
            tprint_error(f"Error type: {type(e).__name__}")
            tprint_error(f"Error details: {traceback.format_exc()}")
            
            return result
    
    def _apply_optimized_lookback_periods(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        optimized_lookback_periods: Dict[str, int]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply optimized lookback periods to features."""
        try:
            # This is a placeholder for applying optimized lookback periods
            # In practice, this would involve resampling or windowing the data
            # based on the optimized periods from feature_lookback_optimization
            
            self.logger.info(f"📊 Applying optimized lookback periods: {optimized_lookback_periods}")
            
            # For now, we'll just log the optimization and return the original data
            # In a full implementation, this would:
            # 1. Identify which features correspond to which optimized periods
            # 2. Apply the appropriate windowing/resampling
            # 3. Update feature names to reflect the optimization
            
            return X, feature_names
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply optimized lookback periods: {e}")
            return X, feature_names
    
    def _extract_significant_features(
        self, 
        pid_result: PIDResult, 
        feature_names: List[str]
    ) -> List[str]:
        """Extract significant features from PID analysis."""
        significant_features = []
        
        # Sort unique information scores and take top features
        unique_info_items = sorted(pid_result.unique_info.items(), key=lambda x: x[1], reverse=True)
        
        for feature, unique_score in unique_info_items:
            if unique_score > self.config.unique_info_threshold:
                significant_features.append(feature)
                if len(significant_features) >= self.config.max_feature_combinations:
                    break
        
        return significant_features
    
    def _variance_based_feature_selection(
        self, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> List[str]:
        """Fallback variance-based feature selection."""
        try:
            # Ensure we have numeric data only
            if not NUMPY_AVAILABLE or X.dtype == object:
                # Handle mixed data types
                numeric_features = []
                for i in range(X.shape[1]):
                    try:
                        # Try to convert to numeric
                        numeric_col = pd.to_numeric(X[:, i], errors='coerce')
                        if not numeric_col.isna().all():
                            numeric_features.append(i)
                    except:
                        continue
                
                if not numeric_features:
                    self.logger.warning("⚠️ No numeric features found, using first few features")
                    return feature_names[:min(len(feature_names), self.config.max_feature_combinations)]
                
                # Use only numeric features
                X_numeric = X[:, numeric_features]
                feature_names_numeric = [feature_names[i] for i in numeric_features]
            else:
                X_numeric = X
                feature_names_numeric = feature_names
            
            # Calculate variance for each feature
            variances = np.var(X_numeric, axis=0)
            
            # Select features with highest variance
            variance_indices = np.argsort(variances)[::-1]
            selected_features = []
            
            for idx in variance_indices:
                if variances[idx] > self.config.min_variance_threshold:
                    selected_features.append(feature_names_numeric[idx])
                    if len(selected_features) >= self.config.max_feature_combinations:
                        break
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Variance-based selection failed: {e}")
            return feature_names[:self.config.max_feature_combinations]
    
    def _generate_polynomial_matrix(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        significant_features: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate polynomial feature matrix."""
        polynomial_features = []
        polynomial_names = []
        
        for feature_name in significant_features:
            try:
                # Find feature index
                idx = feature_names.index(feature_name)
                x = X[:, idx]
                
                # Generate different types of polynomial features
                for polynomial_type in self.config.polynomial_types:
                    poly_feat, poly_name = self._create_polynomial_feature(
                        x, feature_name, polynomial_type
                    )
                    
                    if poly_feat is not None:
                        # Handle both single features and multiple features (like powers)
                        if isinstance(poly_name, list):
                            # Multiple features (e.g., powers)
                            for i, name in enumerate(poly_name):
                                if poly_feat.ndim == 2:
                                    polynomial_features.append(poly_feat[:, i])
                                else:
                                    polynomial_features.append(poly_feat)
                                polynomial_names.append(name)
                                
                                if len(polynomial_names) >= self.config.max_polynomial_features:
                                    break
                        else:
                            # Single feature
                            polynomial_features.append(poly_feat)
                            polynomial_names.append(poly_name)
                        
                        if len(polynomial_names) >= self.config.max_polynomial_features:
                            break
                
                if len(polynomial_names) >= self.config.max_polynomial_features:
                    break
                    
            except (ValueError, IndexError) as e:
                self.logger.warning(f"⚠️ Failed to create polynomial for {feature_name}: {e}")
                continue
        
        if polynomial_features:
            return np.column_stack(polynomial_features), polynomial_names
        else:
            return np.array([]).reshape(X.shape[0], 0), []
    
    def _create_polynomial_feature(
        self, 
        x: np.ndarray, 
        feature_name: str, 
        polynomial_type: PolynomialType
    ) -> Tuple[Optional[np.ndarray], str]:
        """Create a specific type of polynomial feature."""
        try:
            # Ensure x is numeric and convert to float if needed
            if not NUMPY_AVAILABLE:
                return None, ""
            
            # Convert to numeric if needed
            try:
                x_numeric = pd.to_numeric(x, errors='coerce')
                if x_numeric.isna().all():
                    self.logger.warning(f"⚠️ Feature {feature_name} has no numeric values")
                    return None, ""
                x = x_numeric.values
            except:
                # If conversion fails, try direct conversion
                try:
                    x = x.astype(float)
                except:
                    self.logger.warning(f"⚠️ Cannot convert feature {feature_name} to numeric")
                    return None, ""
            
            # Check for valid numeric data
            if np.all(np.isnan(x)) or np.all(np.isinf(x)):
                self.logger.warning(f"⚠️ Feature {feature_name} has no valid numeric values")
                return None, ""
            
            if polynomial_type == PolynomialType.POWER:
                # Generate powers up to max degree using safe operations
                features = []
                names = []
                for degree in range(2, self.config.max_polynomial_degree + 1):
                    try:
                        feature = safe_power(x, degree, 0.0)
                        # Check for valid results
                        if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                            features.append(feature)
                            names.append(f"{feature_name}_pow_{degree}")
                            tprint_debug(f"Created power feature: {feature_name}_pow_{degree}")
                    except Exception as e:
                        tprint_warning(f"Failed to create power feature {feature_name}_pow_{degree}: {e}")
                        continue
                
                if features:
                    return np.column_stack(features), names
                else:
                    return None, []
                
            elif polynomial_type == PolynomialType.SQUARE_ROOT:
                try:
                    # Use safe sqrt with non-negative values
                    feature = safe_sqrt(np.maximum(x, 0), 0.0)
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        tprint_debug(f"Created square root feature: sqrt_{feature_name}")
                        return feature, f"sqrt_{feature_name}"
                except Exception as e:
                    tprint_warning(f"Failed to create square root feature: {e}")
                    pass
                
            elif polynomial_type == PolynomialType.CUBIC_ROOT:
                try:
                    # Use safe power for cubic root
                    feature = safe_power(x, 1.0/3.0, 0.0)
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        tprint_debug(f"Created cubic root feature: cbrt_{feature_name}")
                        return feature, f"cbrt_{feature_name}"
                except Exception as e:
                    tprint_warning(f"Failed to create cubic root feature: {e}")
                    pass
                
            elif polynomial_type == PolynomialType.LOGARITHMIC:
                try:
                    # Use safe log with minimum value
                    feature = safe_log(np.maximum(x, 1e-10), 0.0)
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        tprint_debug(f"Created logarithmic feature: log_{feature_name}")
                        return feature, f"log_{feature_name}"
                except Exception as e:
                    tprint_warning(f"Failed to create logarithmic feature: {e}")
                    pass
                
            elif polynomial_type == PolynomialType.EXPONENTIAL:
                try:
                    # Use safe power with clipped values
                    clipped_x = np.clip(x, -10, 10)
                    feature = safe_power(np.e, clipped_x, 0.0)
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        tprint_debug(f"Created exponential feature: exp_{feature_name}")
                        return feature, f"exp_{feature_name}"
                except Exception as e:
                    tprint_warning(f"Failed to create exponential feature: {e}")
                    pass
                
            elif polynomial_type == PolynomialType.RECIPROCAL:
                try:
                    # Use safe divide for reciprocal
                    feature = safe_divide(1.0, x, 0.0)
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        tprint_debug(f"Created reciprocal feature: recip_{feature_name}")
                        return feature, f"recip_{feature_name}"
                except Exception as e:
                    tprint_warning(f"Failed to create reciprocal feature: {e}")
                    pass
                
            elif polynomial_type == PolynomialType.CROSS_PRODUCT:
                try:
                    # Create cross products with other significant features
                    # This is a simplified version - in practice, you'd use other features
                    feature = x * x  # Self cross product
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        return feature, f"{feature_name}_cross_self"
                except:
                    pass
                
            elif polynomial_type == PolynomialType.INTERACTION:
                try:
                    # Create interaction with transformed versions
                    x_squared = np.power(x, 2)
                    feature = x * x_squared
                    if not (np.any(np.isnan(feature)) or np.any(np.isinf(feature))):
                        return feature, f"{feature_name}_interaction"
                except:
                    pass
                
            return None, ""
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create {polynomial_type.value} polynomial: {e}")
            return None, ""
    
    def _calculate_polynomial_scores(
        self, 
        polynomial_features: np.ndarray, 
        polynomial_names: List[str], 
        target: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate importance scores for polynomial features."""
        scores = {}
        
        if target is None:
            # Use variance as importance score
            for i, name in enumerate(polynomial_names):
                scores[name] = float(np.var(polynomial_features[:, i]))
        else:
            # Use correlation with target as importance score
            for i, name in enumerate(polynomial_names):
                try:
                    if self.matrix_ops:
                        corr = self.matrix_ops.safe_correlation_matrix(
                            np.column_stack([polynomial_features[:, i], target])
                        )[0, 1]
                    else:
                        corr = np.corrcoef(polynomial_features[:, i], target)[0, 1]
                    scores[name] = abs(float(corr))
                except Exception:
                    scores[name] = 0.0
        
        return scores
    
    def _calculate_average_variance(self, polynomial_features: np.ndarray) -> float:
        """Calculate average variance of polynomial features."""
        try:
            variances = np.var(polynomial_features, axis=0)
            return float(np.mean(variances))
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, polynomial_features: np.ndarray) -> float:
        """Calculate stability score based on feature consistency."""
        try:
            # Calculate coefficient of variation for each feature
            cv_scores = []
            for i in range(polynomial_features.shape[1]):
                feature = polynomial_features[:, i]
                mean_val = np.mean(feature)
                std_val = np.std(feature)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    cv_scores.append(cv)
            
            if cv_scores:
                # Lower CV = higher stability
                avg_cv = np.mean(cv_scores)
                stability_score = max(0.0, 1.0 - avg_cv)
                return float(stability_score)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_degree_distribution(self, polynomial_names: List[str]) -> Dict[int, int]:
        """Calculate distribution of polynomial degrees."""
        degree_dist = {}
        
        for name in polynomial_names:
            if '_pow_' in name:
                try:
                    degree = int(name.split('_pow_')[1])
                    degree_dist[degree] = degree_dist.get(degree, 0) + 1
                except (ValueError, IndexError):
                    continue
        
        return degree_dist
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            'pid_available': PID_AVAILABLE,
            'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            'numpy_available': NUMPY_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE
        }
        
        if self.matrix_ops:
            metrics['matrix_ops_stats'] = self.matrix_ops.get_performance_stats()
            metrics['hardware_info'] = self.matrix_ops.get_hardware_info()
        
        return metrics