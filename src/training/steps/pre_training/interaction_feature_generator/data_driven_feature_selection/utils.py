"""
Utility classes and functions for Data-Driven Feature Selection

This module provides utility classes for wrapping feature generators,
estimating costs and utilities, and managing the feature selection process.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

# Import feature bank
try:
    from src.feature_generation.core.feature_bank import get_global_feature_bank, FeatureBank
    from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureCategory
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False
    FeatureGenerator = None
    FeatureCategory = None

# Import matrix operations for cost estimation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


def _log(message: str, level: str = "info") -> None:
    """Centralized logging helper that routes messages through tprint."""
    prefix_map = {
        "info": "📊",
        "debug": "🔍",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
    }

    prefix = prefix_map.get(level, "🔍")
    formatted_message = f"{prefix} [FeatureSelection] {message}"

    if level == "info":
        tprint_info(formatted_message)
    elif level == "warning":
        tprint_warning(formatted_message)
    elif level == "error":
        tprint_error(formatted_message)
    elif level == "success":
        tprint_success(formatted_message)
    else:
        tprint(formatted_message)


def _log_info(message: str) -> None:
    _log(message, level="info")


def _log_debug(message: str) -> None:
    _log(message, level="debug")


def _log_warning(message: str) -> None:
    _log(message, level="warning")


def _log_error(message: str) -> None:
    _log(message, level="error")


def _log_success(message: str) -> None:
    _log(message, level="success")


@dataclass
class FeatureGeneratorWrapper:
    """Wrapper for feature generators with metadata and evaluation capabilities."""
    
    generator: Any  # FeatureGenerator instance
    family: str
    category: str
    name: str
    description: str = ""
    
    # Cost estimates
    estimated_compute_cost_ms: float = 0.0
    estimated_memory_cost_mb: float = 0.0
    estimated_latency_cost_ms: float = 0.0
    total_cost: float = 0.0
    
    # Utility estimates
    estimated_utility: float = 0.0
    utility_uncertainty: float = 1.0
    stability_score: float = 0.0
    
    # Availability constraints
    requires_book_data: bool = False
    requires_tick_data: bool = False
    requires_volume_data: bool = True
    data_availability: float = 1.0
    
    # Evaluation results
    phase1_utility: Optional[float] = None
    phase1_uncertainty: Optional[float] = None
    phase1_stability: Optional[float] = None
    phase2_utility: Optional[float] = None
    phase2_uncertainty: Optional[float] = None
    phase2_stability: Optional[float] = None
    
    # Selection status
    selected_phase1: bool = False
    selected_phase2: bool = False
    selected_final: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        _log_debug(
            f"Serializing wrapper '{self.name}' with total_cost={self.total_cost:.2f}ms "
            f"and estimated_utility={self.estimated_utility:.4f}"
        )
        return {
            'name': self.name,
            'family': self.family,
            'category': self.category,
            'description': self.description,
            'estimated_compute_cost_ms': self.estimated_compute_cost_ms,
            'estimated_memory_cost_mb': self.estimated_memory_cost_mb,
            'estimated_latency_cost_ms': self.estimated_latency_cost_ms,
            'total_cost': self.total_cost,
            'estimated_utility': self.estimated_utility,
            'utility_uncertainty': self.utility_uncertainty,
            'stability_score': self.stability_score,
            'requires_book_data': self.requires_book_data,
            'requires_tick_data': self.requires_tick_data,
            'requires_volume_data': self.requires_volume_data,
            'data_availability': self.data_availability,
            'phase1_utility': self.phase1_utility,
            'phase1_uncertainty': self.phase1_uncertainty,
            'phase1_stability': self.phase1_stability,
            'phase2_utility': self.phase2_utility,
            'phase2_uncertainty': self.phase2_uncertainty,
            'phase2_stability': self.phase2_stability,
            'selected_phase1': self.selected_phase1,
            'selected_phase2': self.selected_phase2,
            'selected_final': self.selected_final
        }


class CostEstimator:
    """Estimates computational costs for feature generators."""
    
    def __init__(self, matrix_ops=None, hardware_processor=None):
        self.matrix_ops = matrix_ops
        self.hardware_processor = hardware_processor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def estimate_generator_cost(self, wrapper: FeatureGeneratorWrapper,
                              data_shape: Tuple[int, int]) -> FeatureGeneratorWrapper:
        """Estimate costs for a feature generator wrapper."""
        try:
            _log_info(
                f"Estimating costs for wrapper '{wrapper.name}' with data_shape={data_shape}"
            )
            # Estimate compute cost based on generator type and data size
            compute_cost = self._estimate_compute_cost(wrapper, data_shape)

            # Estimate memory cost
            memory_cost = self._estimate_memory_cost(wrapper, data_shape)
            
            # Estimate latency cost
            latency_cost = self._estimate_latency_cost(wrapper, data_shape)
            
            # Calculate total cost
            total_cost = compute_cost + memory_cost + latency_cost
            
            # Update wrapper
            wrapper.estimated_compute_cost_ms = compute_cost
            wrapper.estimated_memory_cost_mb = memory_cost
            wrapper.estimated_latency_cost_ms = latency_cost
            wrapper.total_cost = total_cost

            _log_success(
                f"Cost estimation complete for '{wrapper.name}': compute={compute_cost:.2f}ms, "
                f"memory={memory_cost:.2f}MB, latency={latency_cost:.2f}ms, total={total_cost:.2f}"
            )

            return wrapper

        except Exception as e:
            _log_warning(f"Falling back to default cost estimates for '{wrapper.name}': {e}")
            self.logger.warning(f"Failed to estimate cost for {wrapper.name}: {e}")
            # Set default costs
            wrapper.estimated_compute_cost_ms = 10.0
            wrapper.estimated_memory_cost_mb = 1.0
            wrapper.estimated_latency_cost_ms = 5.0
            wrapper.total_cost = 16.0
            return wrapper
    
    def _estimate_compute_cost(self, wrapper: FeatureGeneratorWrapper, 
                             data_shape: Tuple[int, int]) -> float:
        """Estimate compute cost in milliseconds."""
        n_rows, n_cols = data_shape
        
        # Base cost by category
        base_costs = {
            'momentum': 1.0,
            'volatility': 1.2,
            'trend': 1.1,
            'volume': 1.3,
            'oscillator': 1.0,
            'returns': 0.8,
            'support_resistance': 1.5,
            'order_flow': 2.0,
            'microstructure': 2.5,
            'regime': 3.0,
            'entropy': 2.0,
            'acceleration': 1.8,
            'time': 0.5,
            'normalization': 0.8,
            'representation_learning': 4.0
        }
        
        base_cost = base_costs.get(wrapper.category, 1.0)

        # Scale by data size (logarithmic scaling)
        size_factor = np.log(1 + n_rows / 1000) * np.log(1 + n_cols / 10)

        # Scale by complexity
        complexity_factors = {
            'simple': 1.0,
            'medium': 1.5,
            'complex': 2.0,
            'very_complex': 3.0
        }

        complexity = self._assess_complexity(wrapper)
        complexity_factor = complexity_factors.get(complexity, 1.5)

        # Apply hardware acceleration if available
        if self.hardware_processor:
            acceleration_factor = 0.3  # 70% reduction with hardware acceleration
        else:
            acceleration_factor = 1.0

        total_cost = base_cost * size_factor * complexity_factor * acceleration_factor

        _log_debug(
            f"Computed cost for '{wrapper.name}': base={base_cost:.2f}, size_factor={size_factor:.3f}, "
            f"complexity={complexity}({complexity_factor:.2f}), acceleration_factor={acceleration_factor:.2f}, "
            f"result={total_cost:.2f}ms"
        )

        return max(0.1, total_cost)  # Minimum 0.1ms
    
    def _estimate_memory_cost(self, wrapper: FeatureGeneratorWrapper, 
                            data_shape: Tuple[int, int]) -> float:
        """Estimate memory cost in MB."""
        n_rows, n_cols = data_shape
        
        # Base memory by category
        base_memory = {
            'momentum': 0.1,
            'volatility': 0.15,
            'trend': 0.12,
            'volume': 0.2,
            'oscillator': 0.1,
            'returns': 0.05,
            'support_resistance': 0.25,
            'order_flow': 0.5,
            'microstructure': 0.8,
            'regime': 1.0,
            'entropy': 0.6,
            'acceleration': 0.4,
            'time': 0.02,
            'normalization': 0.1,
            'representation_learning': 2.0
        }
        
        base_mb = base_memory.get(wrapper.category, 0.2)
        
        # Scale by data size
        size_factor = n_rows / 10000  # Scale with data size
        
        # Scale by complexity
        complexity = self._assess_complexity(wrapper)
        complexity_factors = {
            'simple': 1.0,
            'medium': 1.5,
            'complex': 2.5,
            'very_complex': 4.0
        }
        complexity_factor = complexity_factors.get(complexity, 1.5)
        
        total_memory = base_mb * size_factor * complexity_factor

        _log_debug(
            f"Estimated memory for '{wrapper.name}': base={base_mb:.3f}MB, size_factor={size_factor:.3f}, "
            f"complexity={complexity}({complexity_factor:.2f}) -> {total_memory:.3f}MB"
        )

        return max(0.01, total_memory)  # Minimum 0.01 MB
    
    def _estimate_latency_cost(self, wrapper: FeatureGeneratorWrapper, 
                             data_shape: Tuple[int, int]) -> float:
        """Estimate latency cost in milliseconds."""
        # Latency is primarily determined by compute cost
        compute_cost = wrapper.estimated_compute_cost_ms
        
        # Add network latency for data-dependent features
        if wrapper.requires_book_data or wrapper.requires_tick_data:
            network_latency = 2.0  # 2ms network latency
        else:
            network_latency = 0.0
        
        # Add I/O latency for complex features
        if wrapper.category in ['regime', 'representation_learning', 'microstructure']:
            io_latency = 1.0  # 1ms I/O latency
        else:
            io_latency = 0.0
        
        total_latency = compute_cost + network_latency + io_latency

        _log_debug(
            f"Estimated latency for '{wrapper.name}': compute={compute_cost:.2f}ms, "
            f"network_latency={network_latency:.2f}ms, io_latency={io_latency:.2f}ms -> {total_latency:.2f}ms"
        )

        return max(0.1, total_latency)  # Minimum 0.1ms

    def _assess_complexity(self, wrapper: FeatureGeneratorWrapper) -> str:
        """Assess the computational complexity of a feature generator."""
        name = wrapper.name.lower()
        category = wrapper.category.lower()

        complexity = 'simple'

        # Very complex features
        if any(term in name for term in ['autoencoder', 'deep', 'neural', 'transformer']):
            complexity = 'very_complex'
        # Complex features
        elif category in ['regime', 'representation_learning', 'microstructure']:
            complexity = 'complex'
        elif any(term in name for term in ['hmm', 'kalman', 'particle', 'monte_carlo']):
            complexity = 'complex'
        # Medium complexity features
        elif category in ['order_flow', 'entropy', 'acceleration']:
            complexity = 'medium'
        elif any(term in name for term in ['fourier', 'wavelet', 'spectral', 'correlation']):
            complexity = 'medium'

        _log_debug(
            f"Complexity assessed for '{wrapper.name}' (category={wrapper.category}) -> {complexity}"
        )

        return complexity


class UtilityEstimator:
    """Estimates utility (predictive value) for feature generators."""
    
    def __init__(self, matrix_ops=None):
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def estimate_phase1_utility(self, wrapper: FeatureGeneratorWrapper, 
                              data: pd.DataFrame, target: np.ndarray) -> FeatureGeneratorWrapper:
        """Estimate utility using Phase 1 cheap probes."""
        try:
            # Generate cheap proxy feature
            proxy_feature = self._generate_cheap_proxy(wrapper, data)
            
            if proxy_feature is None or len(proxy_feature) < 10:
                wrapper.phase1_utility = 0.0
                wrapper.phase1_uncertainty = 1.0
                wrapper.phase1_stability = 0.0
                return wrapper
            
            # Compute IC with block bootstrap
            ic, ic_error = self._compute_ic_with_bootstrap(proxy_feature, target)
            
            # Compute stability score
            stability = self._compute_stability_score(proxy_feature, target)
            
            # Apply cost penalty
            cost_penalty = 0.1 * wrapper.total_cost  # Small penalty for Phase 1
            utility = ic - cost_penalty
            
            # Update wrapper
            wrapper.phase1_utility = utility
            wrapper.phase1_uncertainty = ic_error
            wrapper.phase1_stability = stability
            
            return wrapper
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate Phase 1 utility for {wrapper.name}: {e}")
            wrapper.phase1_utility = 0.0
            wrapper.phase1_uncertainty = 1.0
            wrapper.phase1_stability = 0.0
            return wrapper
    
    def estimate_phase2_utility(self, wrapper: FeatureGeneratorWrapper, 
                              data: pd.DataFrame, target: np.ndarray) -> FeatureGeneratorWrapper:
        """Estimate utility using Phase 2 rich probes with Bayesian optimization."""
        try:
            # This would integrate with the Bayesian lookback optimization system
            # For now, use a more sophisticated version of Phase 1
            
            # Generate multiple proxy features with different lookbacks
            proxy_features = self._generate_rich_proxies(wrapper, data)
            
            if not proxy_features:
                wrapper.phase2_utility = 0.0
                wrapper.phase2_uncertainty = 1.0
                wrapper.phase2_stability = 0.0
                return wrapper
            
            # Compute utilities for all proxies
            utilities = []
            uncertainties = []
            stabilities = []
            
            for proxy_feature in proxy_features:
                if len(proxy_feature) < 10:
                    continue
                
                ic, ic_error = self._compute_ic_with_bootstrap(proxy_feature, target)
                stability = self._compute_stability_score(proxy_feature, target)
                
                utilities.append(ic)
                uncertainties.append(ic_error)
                stabilities.append(stability)
            
            if not utilities:
                wrapper.phase2_utility = 0.0
                wrapper.phase2_uncertainty = 1.0
                wrapper.phase2_stability = 0.0
                return wrapper
            
            # Use best utility (optimistic)
            best_idx = np.argmax(utilities)
            best_utility = utilities[best_idx]
            best_uncertainty = uncertainties[best_idx]
            best_stability = stabilities[best_idx]
            
            # Apply cost penalty
            cost_penalty = 0.2 * wrapper.total_cost  # Higher penalty for Phase 2
            utility = best_utility - cost_penalty
            
            # Update wrapper
            wrapper.phase2_utility = utility
            wrapper.phase2_uncertainty = best_uncertainty
            wrapper.phase2_stability = best_stability
            
            return wrapper
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate Phase 2 utility for {wrapper.name}: {e}")
            wrapper.phase2_utility = 0.0
            wrapper.phase2_uncertainty = 1.0
            wrapper.phase2_stability = 0.0
            return wrapper
    
    def _generate_cheap_proxy(self, wrapper: FeatureGeneratorWrapper,
                            data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate a cheap proxy feature for Phase 1 evaluation."""
        try:
            _log_info(f"Generating cheap proxy for '{wrapper.name}' with minimal lookback")
            # Use the generator with minimal parameters
            if hasattr(wrapper.generator, 'generate'):
                result = wrapper.generator.generate(data, lookback=10)  # Minimal lookback
                if hasattr(result, 'data'):
                    proxy = result.data.values
                    _log_success(
                        f"Cheap proxy generated for '{wrapper.name}' (length={len(proxy)})"
                    )
                    return proxy
                elif isinstance(result, pd.Series):
                    proxy = result.values
                    _log_success(
                        f"Cheap proxy generated for '{wrapper.name}' (length={len(proxy)})"
                    )
                    return proxy
                elif isinstance(result, np.ndarray):
                    proxy = result
                    _log_success(
                        f"Cheap proxy generated for '{wrapper.name}' (length={len(proxy)})"
                    )
                    return proxy
                else:
                    return None
            else:
                return None

        except Exception as e:
            _log_warning(f"Failed to generate cheap proxy for '{wrapper.name}': {e}")
            self.logger.debug(f"Failed to generate cheap proxy for {wrapper.name}: {e}")
            return None

    def _generate_rich_proxies(self, wrapper: FeatureGeneratorWrapper,
                             data: pd.DataFrame) -> List[np.ndarray]:
        """Generate rich proxy features for Phase 2 evaluation."""
        proxies = []

        # Try different lookbacks
        lookbacks = [5, 10, 15, 20, 25]

        _log_info(
            f"Generating rich proxies for '{wrapper.name}' using lookbacks={lookbacks}"
        )

        for lookback in lookbacks:
            try:
                if hasattr(wrapper.generator, 'generate'):
                    result = wrapper.generator.generate(data, lookback=lookback)
                    if hasattr(result, 'data'):
                        proxy = result.data.values
                    elif isinstance(result, pd.Series):
                        proxy = result.values
                    elif isinstance(result, np.ndarray):
                        proxy = result
                    else:
                        continue

                    if len(proxy) > 10:
                        proxies.append(proxy)
                        _log_debug(
                            f"Rich proxy accepted for '{wrapper.name}' at lookback={lookback} (length={len(proxy)})"
                        )

            except Exception as e:
                _log_warning(
                    f"Failed to generate rich proxy for '{wrapper.name}' with lookback={lookback}: {e}"
                )
                self.logger.debug(f"Failed to generate rich proxy for {wrapper.name} with lookback {lookback}: {e}")
                continue

        _log_info(
            f"Generated {len(proxies)} rich proxies for '{wrapper.name}'"
        )

        return proxies

    def _compute_ic_with_bootstrap(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Compute IC with block bootstrap standard errors."""
        try:
            # Remove NaN values
            valid_mask = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_mask) < 10:
                return 0.0, 1.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]

            _log_debug(
                f"Computing IC with bootstrap for proxy (n={len(feature_clean)})"
            )

            # Compute correlation (IC)
            ic = np.corrcoef(feature_clean, target_clean)[0, 1]

            if np.isnan(ic):
                _log_warning("IC computation returned NaN, defaulting to neutral values")
                return 0.0, 1.0

            # Simple bootstrap for standard error
            n_samples = min(100, len(feature_clean) // 4)
            bootstrap_ics = []
            
            for _ in range(n_samples):
                indices = np.random.choice(len(feature_clean), size=len(feature_clean), replace=True)
                f_boot = feature_clean[indices]
                t_boot = target_clean[indices]
                
                if len(np.unique(f_boot)) > 1 and len(np.unique(t_boot)) > 1:
                    boot_ic = np.corrcoef(f_boot, t_boot)[0, 1]
                    if not np.isnan(boot_ic):
                        bootstrap_ics.append(boot_ic)

            if bootstrap_ics:
                ic_error = np.std(bootstrap_ics)
            else:
                ic_error = np.sqrt((1 - ic**2) / (len(feature_clean) - 2))

            _log_debug(
                f"Bootstrap IC results: ic={ic:.4f}, error={ic_error:.4f}, samples={len(bootstrap_ics)}"
            )

            return float(ic), float(ic_error)

        except Exception as e:
            _log_error(f"Failed to compute IC with bootstrap: {e}")
            self.logger.debug(f"Failed to compute IC with bootstrap: {e}")
            return 0.0, 1.0
    
    def _compute_stability_score(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Compute stability score for feature."""
        try:
            # Remove NaN values
            valid_mask = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_mask) < 20:
                return 0.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Split data into thirds
            n = len(feature_clean)
            third = n // 3
            
            # Oldest third
            old_feature = feature_clean[:third]
            old_target = target_clean[:third]
            
            # Newest third
            new_feature = feature_clean[-third:]
            new_target = target_clean[-third:]
            
            # Compute IC for each third
            if len(np.unique(old_feature)) > 1 and len(np.unique(old_target)) > 1:
                old_ic = np.corrcoef(old_feature, old_target)[0, 1]
            else:
                old_ic = 0.0
            
            if len(np.unique(new_feature)) > 1 and len(np.unique(new_target)) > 1:
                new_ic = np.corrcoef(new_feature, new_target)[0, 1]
            else:
                new_ic = 0.0
            
            # Stability score based on consistency
            if np.isnan(old_ic) or np.isnan(new_ic):
                return 0.0
            
            # Higher stability if both ICs have same sign and similar magnitude
            if old_ic * new_ic > 0:  # Same sign
                stability = 1.0 - abs(old_ic - new_ic) / (abs(old_ic) + abs(new_ic) + 1e-6)
            else:  # Different signs
                stability = 0.0
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.debug(f"Failed to compute stability score: {e}")
            return 0.0


def create_feature_generator_wrappers(feature_bank: Optional[FeatureBank] = None) -> List[FeatureGeneratorWrapper]:
    """Create feature generator wrappers from the feature bank."""
    if not FEATURE_BANK_AVAILABLE:
        _log_warning("Feature bank not available, returning empty list")
        return []

    if feature_bank is None:
        try:
            feature_bank = get_global_feature_bank()
            _log_success("Loaded global feature bank for wrapper creation")
        except Exception as e:
            _log_error(f"Failed to get global feature bank: {e}")
            return []

    wrappers = []

    try:
        _log_info("Creating feature generator wrappers from registry entries")
        # Get all generators from the feature bank
        all_generators = feature_bank.registry.get_all()

        for generator in all_generators:
            try:
                # Create wrapper
                wrapper = FeatureGeneratorWrapper(
                    generator=generator,
                    family=generator.config.category.value,
                    category=generator.config.category.value,
                    name=generator.config.name,
                    description=getattr(generator.config, 'description', ''),
                    requires_book_data=_requires_book_data(generator),
                    requires_tick_data=_requires_tick_data(generator),
                    requires_volume_data=_requires_volume_data(generator)
                )

                wrappers.append(wrapper)
                _log_debug(
                    f"Wrapper created for generator '{generator.config.name}' in family '{wrapper.family}'"
                )

            except Exception as e:
                _log_warning(
                    f"Failed to create wrapper for generator '{generator.config.name}': {e}"
                )
                logger.warning(f"Failed to create wrapper for generator {generator.config.name}: {e}")
                continue

        _log_success(f"Created {len(wrappers)} feature generator wrappers")

    except Exception as e:
        _log_error(f"Failed to create feature generator wrappers: {e}")

    return wrappers


def _requires_book_data(generator: Any) -> bool:
    """Check if generator requires book data."""
    name = generator.config.name.lower()
    requires = any(term in name for term in ['bid', 'ask', 'book', 'orderbook', 'depth'])
    if requires:
        _log_debug(f"Generator '{generator.config.name}' requires book data")
    return requires


def _requires_tick_data(generator: Any) -> bool:
    """Check if generator requires tick data."""
    name = generator.config.name.lower()
    requires = any(term in name for term in ['tick', 'trade', 'microstructure'])
    if requires:
        _log_debug(f"Generator '{generator.config.name}' requires tick data")
    return requires


def _requires_volume_data(generator: Any) -> bool:
    """Check if generator requires volume data."""
    name = generator.config.name.lower()
    requires = any(term in name for term in ['volume', 'vol', 'obv', 'vwap'])
    if requires:
        _log_debug(f"Generator '{generator.config.name}' requires volume data")
    return requires


def filter_wrappers_by_availability(wrappers: List[FeatureGeneratorWrapper],
                                  data_availability: Dict[str, float]) -> List[FeatureGeneratorWrapper]:
    """Filter wrappers based on data availability."""
    _log_info(
        f"Filtering {len(wrappers)} wrappers using data availability: {data_availability}"
    )
    filtered = []

    for wrapper in wrappers:
        # Check data availability requirements
        if wrapper.requires_book_data and data_availability.get('book_data', 0.0) < 0.95:
            _log_warning(
                f"Wrapper '{wrapper.name}' removed: insufficient book data availability"
            )
            continue
        if wrapper.requires_tick_data and data_availability.get('tick_data', 0.0) < 0.95:
            _log_warning(
                f"Wrapper '{wrapper.name}' removed: insufficient tick data availability"
            )
            continue
        if wrapper.requires_volume_data and data_availability.get('volume_data', 0.0) < 0.8:
            _log_warning(
                f"Wrapper '{wrapper.name}' removed: insufficient volume data availability"
            )
            continue

        # Update data availability
        if wrapper.requires_book_data:
            wrapper.data_availability = data_availability.get('book_data', 0.0)
        elif wrapper.requires_tick_data:
            wrapper.data_availability = data_availability.get('tick_data', 0.0)
        elif wrapper.requires_volume_data:
            wrapper.data_availability = data_availability.get('volume_data', 0.0)
        else:
            wrapper.data_availability = 1.0

        filtered.append(wrapper)
        _log_debug(
            f"Wrapper '{wrapper.name}' retained with availability={wrapper.data_availability:.2f}"
        )

    _log_info(f"{len(filtered)} wrappers remain after availability filtering")

    return filtered


def compute_feature_correlations(features: pd.DataFrame,
                               threshold: float = 0.9) -> Dict[str, List[str]]:
    """Compute highly correlated feature pairs."""
    _log_info(
        f"Computing feature correlations for {len(features.columns)} features with threshold={threshold}"
        if not features.empty else "No features available for correlation analysis"
    )
    correlations = {}

    if features.empty or len(features.columns) < 2:
        _log_warning("Insufficient features provided for correlation computation")
        return correlations

    # Compute correlation matrix
    corr_matrix = features.corr().abs()

    # Find highly correlated pairs
    pair_count = 0
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.loc[col1, col2] > threshold:
                if col1 not in correlations:
                    correlations[col1] = []
                correlations[col1].append(col2)
                pair_count += 1

    _log_info(f"Identified {pair_count} highly correlated feature pairs")

    return correlations


def apply_diversification_penalty(wrappers: List[FeatureGeneratorWrapper],
                                correlations: Dict[str, List[str]],
                                penalty: float = 0.15) -> List[FeatureGeneratorWrapper]:
    """Apply diversification penalty to highly correlated features."""
    _log_info(
        f"Applying diversification penalty (penalty={penalty}) to {len(wrappers)} wrappers"
    )
    adjusted_wrappers = 0
    for wrapper in wrappers:
        if wrapper.name in correlations:
            # Apply penalty for each correlated feature
            n_correlated = len(correlations[wrapper.name])
            penalty_factor = 1.0 - (penalty * n_correlated)

            # Apply penalty to utilities
            if wrapper.phase1_utility is not None:
                wrapper.phase1_utility *= penalty_factor
            if wrapper.phase2_utility is not None:
                wrapper.phase2_utility *= penalty_factor
            adjusted_wrappers += 1
            _log_debug(
                f"Applied diversification penalty to '{wrapper.name}' with factor={penalty_factor:.2f}"
            )

    _log_info(f"Diversification penalty applied to {adjusted_wrappers} wrappers")

    return wrappers
