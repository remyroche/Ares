"""
Stage 3: Hierarchical Bayesian Shrinkage Across Families and Symbols

This module implements the third stage of the lookback optimization system,
applying hierarchical Bayesian shrinkage to stabilize lookback estimates
across feature families and symbols using variational inference or NUTS sampling.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

# Try to import PyMC for Bayesian inference
try:
    import pymc as pm
    import aesara.tensor as at
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    at = None

# Try to import ArviZ for diagnostics
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    az = None

# Import configuration and previous stage results
from .config import LookbackOptimizationConfig, FamilyType, HierarchicalConfig
from .ic_surface import ICSurfaceResult
from .wf_stability import StabilityResult

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SymbolFamilyData:
    """Data for a single symbol-family combination."""
    symbol: str
    family: FamilyType
    estimated_lookback: float
    lookback_std: float
    ic_value: float
    ic_std: float
    n_observations: int
    stability_score: float = 0.0


@dataclass
class HierarchicalResult:
    """Result of hierarchical Bayesian shrinkage."""
    family_means: Dict[FamilyType, float]
    family_std: Dict[FamilyType, float]
    family_hdi_lower: Dict[FamilyType, float]
    family_hdi_upper: Dict[FamilyType, float]
    shrunk_lookbacks: Dict[Tuple[str, FamilyType], float]
    shrunk_std: Dict[Tuple[str, FamilyType], float]
    shrinkage_factors: Dict[Tuple[str, FamilyType], float]
    convergence_diagnostics: Dict[str, Any]
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'family_means': {k.value: v for k, v in self.family_means.items()},
            'family_std': {k.value: v for k, v in self.family_std.items()},
            'family_hdi_lower': {k.value: v for k, v in self.family_hdi_lower.items()},
            'family_hdi_upper': {k.value: v for k, v in self.family_hdi_upper.items()},
            'shrunk_lookbacks': {f"{k[0]}_{k[1].value}": v for k, v in self.shrunk_lookbacks.items()},
            'shrunk_std': {f"{k[0]}_{k[1].value}": v for k, v in self.shrunk_std.items()},
            'shrinkage_factors': {f"{k[0]}_{k[1].value}": v for k, v in self.shrinkage_factors.items()},
            'convergence_diagnostics': self.convergence_diagnostics,
            'execution_time': self.execution_time
        }


class HierarchicalBayesianShrinkage:
    """Hierarchical Bayesian shrinkage for lookback optimization."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.hierarchical_config = config.hierarchical
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not PYMC_AVAILABLE:
            self.logger.warning("PyMC not available. Hierarchical shrinkage will use fallback methods.")
    
    def apply_shrinkage(self, symbol_family_data: List[SymbolFamilyData]) -> HierarchicalResult:
        """Apply hierarchical Bayesian shrinkage to lookback estimates."""
        start_time = time.time()
        
        try:
            tprint_info("Applying hierarchical Bayesian shrinkage...")
            
            if not PYMC_AVAILABLE:
                return self._fallback_shrinkage(symbol_family_data)
            
            # Group data by family
            family_data = self._group_by_family(symbol_family_data)
            
            if not family_data:
                raise ValueError("No valid family data provided")
            
            # Build hierarchical model
            model = self._build_hierarchical_model(family_data)
            
            # Sample from posterior
            if self.hierarchical_config.use_variational:
                trace = self._variational_inference(model)
            else:
                trace = self._nuts_sampling(model)
            
            # Extract results
            result = self._extract_results(trace, family_data, symbol_family_data)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            tprint_info(f"Hierarchical shrinkage completed in {execution_time:.3f}s")
            tprint_info(f"Processed {len(symbol_family_data)} symbol-family combinations")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Hierarchical shrinkage failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return fallback result
            return self._fallback_shrinkage(symbol_family_data)
    
    def _group_by_family(self, symbol_family_data: List[SymbolFamilyData]) -> Dict[FamilyType, List[SymbolFamilyData]]:
        """Group data by feature family."""
        family_data = {}
        
        for data in symbol_family_data:
            if data.family not in family_data:
                family_data[data.family] = []
            family_data[data.family].append(data)
        
        return family_data
    
    def _build_hierarchical_model(self, family_data: Dict[FamilyType, List[SymbolFamilyData]]) -> 'pm.Model':
        """Build hierarchical Bayesian model."""
        with pm.Model() as model:
            # Family-level parameters
            family_means = {}
            family_taus = {}
            
            for family, data_list in family_data.items():
                # Get lookback estimates and standard errors
                lookbacks = np.array([d.estimated_lookback for d in data_list])
                lookback_stds = np.array([d.lookback_std for d in data_list])
                
                # Transform to log space for better numerical stability
                log_lookbacks = np.log(lookbacks)
                log_lookback_stds = lookback_stds / lookbacks  # Approximate log-space std
                
                # Family-level priors
                family_means[family] = pm.Normal(
                    f"mu_{family.value}",
                    mu=self.hierarchical_config.mu_prior_mean,
                    sigma=self.hierarchical_config.mu_prior_std
                )
                
                family_taus[family] = pm.HalfNormal(
                    f"tau_{family.value}",
                    sigma=self.hierarchical_config.tau_prior_scale
                )
                
                # Symbol-level parameters (non-centered parametrization)
                z = pm.Normal(f"z_{family.value}", 0, 1, shape=len(data_list))
                
                # True lookbacks
                true_log_lookbacks = family_means[family] + family_taus[family] * z
                
                # Likelihood
                pm.Normal(
                    f"obs_{family.value}",
                    mu=true_log_lookbacks,
                    sigma=log_lookback_stds,
                    observed=log_lookbacks
                )
            
            return model
    
    def _variational_inference(self, model: 'pm.Model') -> Any:
        """Perform variational inference using ADVI."""
        try:
            with model:
                # Initialize variational approximation
                approx = pm.ADVI()
                
                # Fit the approximation
                approx.fit(
                    n=self.hierarchical_config.n_samples,
                    progressbar=False
                )
                
                # Sample from the approximation
                trace = approx.sample(self.hierarchical_config.n_samples)
                
                return trace
                
        except Exception as e:
            self.logger.warning(f"Variational inference failed: {e}. Falling back to NUTS.")
            return self._nuts_sampling(model)
    
    def _nuts_sampling(self, model: 'pm.Model') -> Any:
        """Perform NUTS sampling."""
        with model:
            # Sample from posterior
            trace = pm.sample(
                draws=self.hierarchical_config.n_samples,
                tune=self.hierarchical_config.n_tuning,
                target_accept=self.hierarchical_config.target_accept,
                max_treedepth=self.hierarchical_config.max_treedepth,
                adapt_delta=self.hierarchical_config.adapt_delta,
                progressbar=False,
                return_inferencedata=True
            )
            
            return trace
    
    def _extract_results(self, trace: Any, family_data: Dict[FamilyType, List[SymbolFamilyData]], 
                        symbol_family_data: List[SymbolFamilyData]) -> HierarchicalResult:
        """Extract results from MCMC trace."""
        try:
            # Convert to ArviZ format if available
            if ARVIZ_AVAILABLE and hasattr(trace, 'posterior'):
                data = trace
            else:
                data = trace
            
            # Extract family-level parameters
            family_means = {}
            family_std = {}
            family_hdi_lower = {}
            family_hdi_upper = {}
            
            for family in family_data.keys():
                mu_samples = data.posterior[f"mu_{family.value}"].values.flatten()
                
                family_means[family] = float(np.mean(mu_samples))
                family_std[family] = float(np.std(mu_samples))
                
                # Compute HDI
                hdi_lower = np.percentile(mu_samples, 2.5)
                hdi_upper = np.percentile(mu_samples, 97.5)
                family_hdi_lower[family] = float(hdi_lower)
                family_hdi_upper[family] = float(hdi_upper)
            
            # Extract symbol-level shrunk estimates
            shrunk_lookbacks = {}
            shrunk_std = {}
            shrinkage_factors = {}
            
            for data in symbol_family_data:
                symbol_family_key = (data.symbol, data.family)
                
                # Get posterior samples for this symbol-family combination
                if f"z_{data.family.value}" in data.posterior:
                    z_samples = data.posterior[f"z_{data.family.value}"].values.flatten()
                    family_mu = family_means[data.family]
                    family_tau = family_std[data.family]  # Approximate tau from std
                    
                    # Compute shrunk estimates
                    shrunk_log_lookbacks = family_mu + family_tau * z_samples
                    shrunk_lookbacks[symbol_family_key] = float(np.exp(np.mean(shrunk_log_lookbacks)))
                    shrunk_std[symbol_family_key] = float(np.exp(np.std(shrunk_log_lookbacks)))
                    
                    # Compute shrinkage factor
                    original_std = data.lookback_std
                    shrunk_std_val = shrunk_std[symbol_family_key]
                    shrinkage_factor = 1.0 - (shrunk_std_val / original_std) if original_std > 0 else 0.0
                    shrinkage_factors[symbol_family_key] = float(shrinkage_factor)
                else:
                    # Fallback to original estimates
                    shrunk_lookbacks[symbol_family_key] = data.estimated_lookback
                    shrunk_std[symbol_family_key] = data.lookback_std
                    shrinkage_factors[symbol_family_key] = 0.0
            
            # Compute convergence diagnostics
            convergence_diagnostics = self._compute_convergence_diagnostics(data)
            
            return HierarchicalResult(
                family_means=family_means,
                family_std=family_std,
                family_hdi_lower=family_hdi_lower,
                family_hdi_upper=family_hdi_upper,
                shrunk_lookbacks=shrunk_lookbacks,
                shrunk_std=shrunk_std,
                shrinkage_factors=shrinkage_factors,
                convergence_diagnostics=convergence_diagnostics
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to extract results from trace: {e}")
            return self._fallback_shrinkage(symbol_family_data)
    
    def _compute_convergence_diagnostics(self, trace: Any) -> Dict[str, Any]:
        """Compute convergence diagnostics."""
        diagnostics = {}
        
        try:
            if ARVIZ_AVAILABLE:
                # Use ArviZ for comprehensive diagnostics
                summary = az.summary(trace)
                diagnostics['rhat_max'] = float(summary['r_hat'].max())
                diagnostics['rhat_min'] = float(summary['r_hat'].min())
                diagnostics['effective_sample_size_min'] = float(summary['ess_bulk'].min())
                diagnostics['effective_sample_size_max'] = float(summary['ess_bulk'].max())
            else:
                # Basic diagnostics without ArviZ
                diagnostics['rhat_max'] = 1.0  # Placeholder
                diagnostics['rhat_min'] = 1.0
                diagnostics['effective_sample_size_min'] = 1000
                diagnostics['effective_sample_size_max'] = 1000
                
        except Exception as e:
            self.logger.warning(f"Failed to compute convergence diagnostics: {e}")
            diagnostics = {'error': str(e)}
        
        return diagnostics
    
    def _fallback_shrinkage(self, symbol_family_data: List[SymbolFamilyData]) -> HierarchicalResult:
        """Fallback shrinkage when PyMC is not available."""
        tprint_warning("Using fallback shrinkage method (PyMC not available)")
        
        # Group by family
        family_data = self._group_by_family(symbol_family_data)
        
        # Simple empirical Bayes shrinkage
        family_means = {}
        family_std = {}
        family_hdi_lower = {}
        family_hdi_upper = {}
        shrunk_lookbacks = {}
        shrunk_std = {}
        shrinkage_factors = {}
        
        for family, data_list in family_data.items():
            lookbacks = np.array([d.estimated_lookback for d in data_list])
            lookback_stds = np.array([d.lookback_std for d in data_list])
            
            # Family-level statistics
            family_mean = float(np.mean(lookbacks))
            family_std_val = float(np.std(lookbacks))
            
            family_means[family] = family_mean
            family_std[family] = family_std_val
            family_hdi_lower[family] = family_mean - 1.96 * family_std_val
            family_hdi_upper[family] = family_mean + 1.96 * family_std_val
            
            # Shrink individual estimates toward family mean
            for data in data_list:
                symbol_family_key = (data.symbol, data.family)
                
                # Simple shrinkage formula
                shrinkage_weight = 1.0 / (1.0 + (data.lookback_std / family_std_val)**2)
                shrunk_lookback = shrinkage_weight * family_mean + (1 - shrinkage_weight) * data.estimated_lookback
                
                shrunk_lookbacks[symbol_family_key] = shrunk_lookback
                shrunk_std[symbol_family_key] = data.lookback_std * np.sqrt(1 - shrinkage_weight)
                shrinkage_factors[symbol_family_key] = shrinkage_weight
        
        return HierarchicalResult(
            family_means=family_means,
            family_std=family_std,
            family_hdi_lower=family_hdi_lower,
            family_hdi_upper=family_hdi_upper,
            shrunk_lookbacks=shrunk_lookbacks,
            shrunk_std=shrunk_std,
            shrinkage_factors=shrinkage_factors,
            convergence_diagnostics={'method': 'fallback_empirical_bayes'}
        )


class MultiSymbolHierarchicalShrinkage:
    """Apply hierarchical shrinkage across multiple symbols."""
    
    def __init__(self, config: LookbackOptimizationConfig):
        self.config = config
        self.shrinkage = HierarchicalBayesianShrinkage(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def apply_multi_symbol_shrinkage(self, 
                                   ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                                   stability_results: Dict[str, Dict[FamilyType, StabilityResult]]) -> Dict[str, HierarchicalResult]:
        """Apply hierarchical shrinkage across multiple symbols."""
        results = {}
        
        # Collect all symbol-family data
        all_symbol_family_data = []
        
        for symbol, symbol_ic_results in ic_surface_results.items():
            symbol_stability_results = stability_results.get(symbol, {})
            
            for family, ic_result in symbol_ic_results.items():
                stability_result = symbol_stability_results.get(family)
                
                # Create symbol-family data
                data = SymbolFamilyData(
                    symbol=symbol,
                    family=family,
                    estimated_lookback=ic_result.optimal_lookback,
                    lookback_std=ic_result.optimal_ic_error,
                    ic_value=ic_result.optimal_ic,
                    ic_std=ic_result.optimal_ic_error,
                    n_observations=len(ic_result.lookbacks),
                    stability_score=stability_result.stability_score if stability_result else 0.0
                )
                
                all_symbol_family_data.append(data)
        
        if not all_symbol_family_data:
            self.logger.warning("No symbol-family data provided for hierarchical shrinkage")
            return results
        
        # Apply hierarchical shrinkage
        try:
            hierarchical_result = self.shrinkage.apply_shrinkage(all_symbol_family_data)
            
            # For now, return the same result for all symbols
            # In practice, you might want to extract symbol-specific results
            for symbol in ic_surface_results.keys():
                results[symbol] = hierarchical_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-symbol hierarchical shrinkage failed: {e}")
            return results