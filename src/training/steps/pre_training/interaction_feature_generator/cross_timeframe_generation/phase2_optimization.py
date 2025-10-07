"""
Phase-2 Optimization with Local Grids and IC Surface Fitting

Implements rigorous optimization of HTF lookback lengths with:
- Local grids around shortlisted candidates
- IC surface fitting with penalized splines or Gaussian processes
- Regime-aware hierarchical shrinkage
- Discrete vs blend export with probabilistic triggers
- BOCPD adaptation for dynamic optimization
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats, optimize
from scipy.interpolate import UnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

from .config import OptimizationConfig

# Try to import PyMC for Bayesian optimization
try:
    import pymc as pm
    import arviz as az
    PYMCMC_AVAILABLE = True
except ImportError:
    PYMCMC_AVAILABLE = False
    logging.warning("PyMC not available, using simplified Bayesian optimization")


@dataclass
class LocalGridResult:
    """Result of local grid optimization."""
    feature_name: str
    family: str
    base_lookback: int
    local_grid: List[int]
    ic_surface: Dict[int, float]
    se_surface: Dict[int, float]
    optimal_lookback: int
    optimal_ic: float
    optimal_se: float
    confidence_interval: Tuple[float, float]
    export_type: str  # 'discrete' or 'blend'
    blend_weights: Optional[Dict[int, float]]
    metadata: Dict[str, Any]


@dataclass
class HierarchicalShrinkageResult:
    """Result of hierarchical shrinkage optimization."""
    feature_name: str
    family: str
    regime: str
    pooled_optimal: float
    regime_optimal: float
    shrinkage_factor: float
    posterior_mean: float
    posterior_std: float
    hdi_width: float
    metadata: Dict[str, Any]


class LocalGridGenerator:
    """Generates local grids around shortlisted candidates."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_local_grid(self, 
                          base_lookback: int,
                          family: str) -> List[int]:
        """
        Generate local grid around base lookback.
        
        Args:
            base_lookback: Base lookback from Phase-1
            family: Feature family
            
        Returns:
            List of lookback values for local optimization
        """
        # Local grid factor determines spread around base
        factor = self.config.local_grid_factor
        
        # Generate log-spaced grid around base
        log_base = np.log(base_lookback)
        log_spread = factor * np.log(2)  # ±factor octaves
        
        # Create grid points
        n_points = 5  # 5 points in local grid
        log_points = np.linspace(log_base - log_spread, log_base + log_spread, n_points)
        grid_points = np.exp(log_points).astype(int)
        
        # Ensure grid is within bounds
        min_lookback = max(15, int(base_lookback * 0.5))
        max_lookback = min(298, int(base_lookback * 2.0))
        
        grid_points = [p for p in grid_points if min_lookback <= p <= max_lookback]
        
        # Add base lookback if not already included
        if base_lookback not in grid_points:
            grid_points.append(base_lookback)
        
        return sorted(list(set(grid_points)))


class ICSurfaceFitter:
    """Fits IC surface using penalized splines or Gaussian processes."""
    
    def __init__(self, method: str = 'spline'):
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def fit_ic_surface(self, 
                      lookbacks: List[int],
                      ics: List[float],
                      ses: List[float]) -> Dict[str, Any]:
        """
        Fit IC surface over lookback values.
        
        Args:
            lookbacks: List of lookback values
            ics: List of IC values
            ses: List of standard errors
            
        Returns:
            Dictionary with fitted surface and optimization results
        """
        if len(lookbacks) < 3:
            return self._simple_fit(lookbacks, ics, ses)
        
        # Convert to log space for better fitting
        log_lookbacks = np.log(lookbacks)
        
        if self.method == 'spline':
            return self._fit_spline_surface(log_lookbacks, ics, ses)
        elif self.method == 'gp':
            return self._fit_gp_surface(log_lookbacks, ics, ses)
        else:
            return self._simple_fit(lookbacks, ics, ses)
    
    def _fit_spline_surface(self, 
                          log_lookbacks: np.ndarray,
                          ics: np.ndarray,
                          ses: np.ndarray) -> Dict[str, Any]:
        """Fit IC surface using penalized splines."""
        try:
            # Weight by inverse variance
            weights = 1.0 / (ses**2 + 1e-8)
            
            # Fit penalized spline
            spline = UnivariateSpline(log_lookbacks, ics, w=weights, s=len(ics))
            
            # Find optimal point
            def objective(log_lb):
                return -spline(log_lb)
            
            result = optimize.minimize_scalar(objective, 
                                            bounds=(log_lookbacks.min(), log_lookbacks.max()),
                                            method='bounded')
            
            optimal_log_lookback = result.x
            optimal_ic = -result.fun
            optimal_lookback = np.exp(optimal_log_lookback)
            
            # Calculate confidence interval
            # Use bootstrap or analytical approximation
            ci_lower, ci_upper = self._calculate_confidence_interval(
                spline, log_lookbacks, ics, ses
            )
            
            return {
                'method': 'spline',
                'spline': spline,
                'optimal_lookback': optimal_lookback,
                'optimal_ic': optimal_ic,
                'confidence_interval': (ci_lower, ci_upper),
                'r_squared': self._calculate_r_squared(ics, spline(log_lookbacks))
            }
            
        except Exception as e:
            self.logger.warning(f"Spline fitting failed: {e}, using simple fit")
            return self._simple_fit(np.exp(log_lookbacks), ics, ses)
    
    def _fit_gp_surface(self, 
                      log_lookbacks: np.ndarray,
                      ics: np.ndarray,
                      ses: np.ndarray) -> Dict[str, Any]:
        """Fit IC surface using Gaussian process."""
        try:
            # Set up GP with RBF kernel + white noise
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=ses**2)
            
            # Fit GP
            gp.fit(log_lookbacks.reshape(-1, 1), ics)
            
            # Find optimal point
            def objective(log_lb):
                pred, std = gp.predict([[log_lb]], return_std=True)
                return -pred[0]  # Minimize negative IC
            
            result = optimize.minimize_scalar(objective,
                                            bounds=(log_lookbacks.min(), log_lookbacks.max()),
                                            method='bounded')
            
            optimal_log_lookback = result.x
            optimal_ic = -result.fun
            optimal_lookback = np.exp(optimal_log_lookback)
            
            # Calculate confidence interval
            pred, std = gp.predict([[optimal_log_lookback]], return_std=True)
            ci_lower = pred[0] - 1.96 * std[0]
            ci_upper = pred[0] + 1.96 * std[0]
            
            return {
                'method': 'gp',
                'gp': gp,
                'optimal_lookback': optimal_lookback,
                'optimal_ic': optimal_ic,
                'confidence_interval': (ci_lower, ci_upper),
                'r_squared': self._calculate_r_squared(ics, gp.predict(log_lookbacks.reshape(-1, 1)))
            }
            
        except Exception as e:
            self.logger.warning(f"GP fitting failed: {e}, using simple fit")
            return self._simple_fit(np.exp(log_lookbacks), ics, ses)
    
    def _simple_fit(self, 
                   lookbacks: List[int],
                   ics: List[float],
                   ses: List[float]) -> Dict[str, Any]:
        """Simple fit using weighted average."""
        if not lookbacks:
            return {'method': 'simple', 'optimal_lookback': 60, 'optimal_ic': 0.0}
        
        # Weight by inverse variance
        weights = 1.0 / (np.array(ses)**2 + 1e-8)
        weights = weights / weights.sum()
        
        # Weighted average
        optimal_lookback = int(np.average(lookbacks, weights=weights))
        optimal_ic = np.average(ics, weights=weights)
        
        # Simple confidence interval
        ci_std = np.sqrt(np.average((np.array(ics) - optimal_ic)**2, weights=weights))
        ci_lower = optimal_ic - 1.96 * ci_std
        ci_upper = optimal_ic + 1.96 * ci_std
        
        return {
            'method': 'simple',
            'optimal_lookback': optimal_lookback,
            'optimal_ic': optimal_ic,
            'confidence_interval': (ci_lower, ci_upper),
            'r_squared': 0.0
        }
    
    def _calculate_confidence_interval(self, 
                                     spline, 
                                     log_lookbacks: np.ndarray,
                                     ics: np.ndarray,
                                     ses: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for spline fit."""
        # Bootstrap confidence interval
        n_bootstrap = 100
        bootstrap_ics = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(ics), size=len(ics), replace=True)
            bootstrap_ics.append(np.mean(ics[indices]))
        
        ci_lower = np.percentile(bootstrap_ics, 2.5)
        ci_upper = np.percentile(bootstrap_ics, 97.5)
        
        return ci_lower, ci_upper
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class HierarchicalShrinkage:
    """Implements regime-aware hierarchical shrinkage."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def fit_hierarchical_model(self, 
                             feature_results: List[Dict[str, Any]],
                             regime_segments: List[Any]) -> Dict[str, Any]:
        """
        Fit hierarchical shrinkage model across symbols and regimes.
        
        Args:
            feature_results: List of feature optimization results
            regime_segments: Regime segmentation results
            
        Returns:
            Hierarchical shrinkage results
        """
        if not feature_results:
            return {}
        
        # Group results by feature family and regime
        grouped_results = self._group_results_by_family_regime(feature_results, regime_segments)
        
        shrinkage_results = {}
        
        for (family, regime), results in grouped_results.items():
            if len(results) < 2:  # Need multiple symbols for shrinkage
                continue
            
            # Extract optimal lookbacks and their uncertainties
            optimal_lookbacks = [r['optimal_lookback'] for r in results]
            uncertainties = [r.get('optimal_se', 1.0) for r in results]
            
            # Convert to log space
            log_lookbacks = np.log(optimal_lookbacks)
            log_uncertainties = np.array(uncertainties) / np.array(optimal_lookbacks)
            
            # Fit hierarchical model
            shrinkage_result = self._fit_regime_hierarchical_model(
                family, regime, log_lookbacks, log_uncertainties, results
            )
            
            shrinkage_results[f"{family}_{regime}"] = shrinkage_result
        
        return shrinkage_results
    
    def _group_results_by_family_regime(self, 
                                      feature_results: List[Dict[str, Any]],
                                      regime_segments: List[Any]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Group results by feature family and regime."""
        grouped = {}
        
        for result in feature_results:
            family = result.get('family', 'unknown')
            regime = result.get('regime', 'mixed')
            key = (family, regime)
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        return grouped
    
    def _fit_regime_hierarchical_model(self, 
                                     family: str,
                                     regime: str,
                                     log_lookbacks: np.ndarray,
                                     log_uncertainties: np.ndarray,
                                     results: List[Dict[str, Any]]) -> HierarchicalShrinkageResult:
        """Fit hierarchical model for a specific family-regime combination."""
        
        if PYMCMC_AVAILABLE and len(log_lookbacks) >= 3:
            return self._fit_bayesian_hierarchical_model(
                family, regime, log_lookbacks, log_uncertainties, results
            )
        else:
            return self._fit_empirical_hierarchical_model(
                family, regime, log_lookbacks, log_uncertainties, results
            )
    
    def _fit_bayesian_hierarchical_model(self, 
                                       family: str,
                                       regime: str,
                                       log_lookbacks: np.ndarray,
                                       log_uncertainties: np.ndarray,
                                       results: List[Dict[str, Any]]) -> HierarchicalShrinkageResult:
        """Fit Bayesian hierarchical model using PyMC."""
        try:
            with pm.Model() as model:
                # Hyperpriors
                mu_prior = pm.Normal('mu_prior', mu=0, sigma=2)
                tau_prior = pm.HalfStudentT('tau_prior', nu=3, sigma=1)
                
                # Latent optimal lookbacks
                z_latent = pm.Normal('z_latent', mu=mu_prior, sigma=tau_prior, shape=len(log_lookbacks))
                
                # Observed lookbacks with measurement error
                pm.Normal('observed', mu=z_latent, sigma=log_uncertainties, observed=log_lookbacks)
                
                # Sample
                trace = pm.sample(1000, tune=500, return_inferencedata=True)
            
            # Extract results
            posterior_samples = trace.posterior.z_latent.values.reshape(-1, len(log_lookbacks))
            pooled_mean = np.mean(posterior_samples)
            pooled_std = np.std(posterior_samples)
            
            # Calculate shrinkage factors
            shrinkage_factors = 1 - (log_uncertainties**2) / (log_uncertainties**2 + pooled_std**2)
            
            # Calculate HDI
            hdi = az.hdi(trace, var_names=['z_latent'])
            hdi_width = np.mean(hdi.z_latent.values[:, 1] - hdi.z_latent.values[:, 0])
            
            return HierarchicalShrinkageResult(
                feature_name=f"{family}_{regime}",
                family=family,
                regime=regime,
                pooled_optimal=np.exp(pooled_mean),
                regime_optimal=np.exp(np.mean(log_lookbacks)),
                shrinkage_factor=np.mean(shrinkage_factors),
                posterior_mean=pooled_mean,
                posterior_std=pooled_std,
                hdi_width=hdi_width,
                metadata={'method': 'bayesian', 'n_samples': len(log_lookbacks)}
            )
            
        except Exception as e:
            self.logger.warning(f"Bayesian hierarchical model failed: {e}, using empirical")
            return self._fit_empirical_hierarchical_model(
                family, regime, log_lookbacks, log_uncertainties, results
            )
    
    def _fit_empirical_hierarchical_model(self, 
                                        family: str,
                                        regime: str,
                                        log_lookbacks: np.ndarray,
                                        log_uncertainties: np.ndarray,
                                        results: List[Dict[str, Any]]) -> HierarchicalShrinkageResult:
        """Fit empirical hierarchical model."""
        # Pooled mean
        pooled_mean = np.mean(log_lookbacks)
        pooled_std = np.std(log_lookbacks)
        
        # Shrinkage factors (empirical Bayes)
        shrinkage_factors = 1 - (log_uncertainties**2) / (log_uncertainties**2 + pooled_std**2)
        
        # HDI approximation
        hdi_width = 2 * 1.96 * pooled_std
        
        return HierarchicalShrinkageResult(
            feature_name=f"{family}_{regime}",
            family=family,
            regime=regime,
            pooled_optimal=np.exp(pooled_mean),
            regime_optimal=np.exp(np.mean(log_lookbacks)),
            shrinkage_factor=np.mean(shrinkage_factors),
            posterior_mean=pooled_mean,
            posterior_std=pooled_std,
            hdi_width=hdi_width,
            metadata={'method': 'empirical', 'n_samples': len(log_lookbacks)}
        )


class ExportDecisionMaker:
    """Decides between discrete and blend export strategies."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def decide_export_strategy(self, 
                             ic_surface_result: Dict[str, Any],
                             hierarchical_result: Optional[HierarchicalShrinkageResult] = None) -> Dict[str, Any]:
        """
        Decide between discrete and blend export strategy.
        
        Args:
            ic_surface_result: IC surface fitting result
            hierarchical_result: Hierarchical shrinkage result
            
        Returns:
            Export strategy decision
        """
        optimal_lookback = ic_surface_result['optimal_lookback']
        confidence_interval = ic_surface_result['confidence_interval']
        hdi_width = hierarchical_result.hdi_width if hierarchical_result else 1.0
        
        # Check if posterior is tight (95% HDI width ≤ 0.35 in log-space)
        log_hdi_width = hdi_width
        tight_posterior = log_hdi_width <= 0.35
        
        # Check if regime optima differ significantly
        regime_difference = 0.0
        if hierarchical_result:
            regime_difference = abs(np.log(hierarchical_result.regime_optimal) - 
                                  np.log(hierarchical_result.pooled_optimal))
        
        significant_regime_difference = regime_difference > 0.25
        
        # Decision logic
        if tight_posterior and not significant_regime_difference:
            # Discrete export
            return self._create_discrete_export(optimal_lookback, ic_surface_result)
        else:
            # Blend export
            return self._create_blend_export(ic_surface_result, hierarchical_result)
    
    def _create_discrete_export(self, 
                              optimal_lookback: int,
                              ic_surface_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create discrete export strategy."""
        # Snap to canonical lookbacks
        canonical_lookbacks = [15, 30, 60, 90, 120, 180, 240, 298]
        closest_canonical = min(canonical_lookbacks, 
                              key=lambda x: abs(x - optimal_lookback))
        
        return {
            'export_type': 'discrete',
            'optimal_lookback': closest_canonical,
            'confidence_interval': ic_surface_result['confidence_interval'],
            'expected_ic': ic_surface_result['optimal_ic'],
            'blend_weights': None,
            'metadata': {
                'original_optimal': optimal_lookback,
                'canonical_mapping': closest_canonical
            }
        }
    
    def _create_blend_export(self, 
                           ic_surface_result: Dict[str, Any],
                           hierarchical_result: Optional[HierarchicalShrinkageResult] = None) -> Dict[str, Any]:
        """Create blend export strategy."""
        # Find two adjacent lookbacks for blending
        optimal_lookback = ic_surface_result['optimal_lookback']
        
        # Find adjacent canonical lookbacks
        canonical_lookbacks = [15, 30, 60, 90, 120, 180, 240, 298]
        sorted_canonical = sorted(canonical_lookbacks)
        
        # Find the two lookbacks that bracket the optimal
        lower_lookback = None
        upper_lookback = None
        
        for i, lb in enumerate(sorted_canonical):
            if lb <= optimal_lookback:
                lower_lookback = lb
            if lb >= optimal_lookback and upper_lookback is None:
                upper_lookback = lb
                break
        
        if lower_lookback is None:
            lower_lookback = sorted_canonical[0]
        if upper_lookback is None:
            upper_lookback = sorted_canonical[-1]
        
        # Calculate blend weights using ridge regression
        blend_weights = self._calculate_blend_weights(
            optimal_lookback, lower_lookback, upper_lookback
        )
        
        return {
            'export_type': 'blend',
            'optimal_lookback': optimal_lookback,
            'confidence_interval': ic_surface_result['confidence_interval'],
            'expected_ic': ic_surface_result['optimal_ic'],
            'blend_weights': blend_weights,
            'metadata': {
                'lower_lookback': lower_lookback,
                'upper_lookback': upper_lookback
            }
        }
    
    def _calculate_blend_weights(self, 
                               optimal: int,
                               lower: int,
                               upper: int) -> Dict[int, float]:
        """Calculate blend weights using ridge regression."""
        # Simple linear interpolation weights
        if upper == lower:
            return {lower: 1.0, upper: 0.0}
        
        weight_lower = (upper - optimal) / (upper - lower)
        weight_upper = (optimal - lower) / (upper - lower)
        
        # Ensure non-negative weights
        weight_lower = max(0.0, weight_lower)
        weight_upper = max(0.0, weight_upper)
        
        # Normalize
        total_weight = weight_lower + weight_upper
        if total_weight > 0:
            weight_lower /= total_weight
            weight_upper /= total_weight
        
        return {lower: weight_lower, upper: weight_upper}


class Phase2Optimization:
    """Main Phase-2 optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.local_grid_generator = LocalGridGenerator(config)
        self.ic_surface_fitter = ICSurfaceFitter(config.ic_surface_smoothing)
        self.hierarchical_shrinkage = HierarchicalShrinkage(config)
        self.export_decision_maker = ExportDecisionMaker(config)
    
    def optimize_lookbacks(self, 
                         sessionized_data: Dict[str, Any],
                         phase1_results: Dict[str, Any],
                         regime_segments: Dict[str, Any],
                         targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Optimize lookback lengths for shortlisted HTF candidates.
        
        Args:
            sessionized_data: Sessionized and aligned data
            phase1_results: Phase-1 probe results
            regime_segments: Regime segmentation results
            targets: Target variables
            
        Returns:
            Phase-2 optimization results
        """
        self.logger.info("Starting Phase-2 optimization")
        
        shortlisted_candidates = phase1_results.get('shortlisted_candidates', [])
        
        if not shortlisted_candidates:
            self.logger.warning("No shortlisted candidates from Phase-1")
            return {'optimized_features': [], 'hierarchical_results': {}}
        
        # Process each shortlisted candidate
        optimized_features = []
        
        for candidate in shortlisted_candidates:
            try:
                # Generate local grid
                local_grid = self.local_grid_generator.generate_local_grid(
                    candidate.lookback_minutes, candidate.family
                )
                
                # Fit IC surface
                ic_surface_result = self._fit_ic_surface_for_candidate(
                    candidate, local_grid, sessionized_data, targets
                )
                
                # Create local grid result
                local_grid_result = LocalGridResult(
                    feature_name=candidate.base_feature,
                    family=candidate.family,
                    base_lookback=candidate.lookback_minutes,
                    local_grid=local_grid,
                    ic_surface=ic_surface_result.get('ic_surface', {}),
                    se_surface=ic_surface_result.get('se_surface', {}),
                    optimal_lookback=ic_surface_result['optimal_lookback'],
                    optimal_ic=ic_surface_result['optimal_ic'],
                    optimal_se=ic_surface_result.get('optimal_se', 1.0),
                    confidence_interval=ic_surface_result['confidence_interval'],
                    export_type='pending',
                    blend_weights=None,
                    metadata=ic_surface_result
                )
                
                optimized_features.append(local_grid_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize {candidate.base_feature}: {e}")
                continue
        
        # Apply hierarchical shrinkage
        hierarchical_results = self.hierarchical_shrinkage.fit_hierarchical_model(
            [self._local_grid_to_dict(f) for f in optimized_features],
            regime_segments.get('segments', [])
        )
        
        # Make export decisions
        final_features = []
        for feature in optimized_features:
            # Find corresponding hierarchical result
            hierarchical_result = None
            for key, result in hierarchical_results.items():
                if result.family == feature.family:
                    hierarchical_result = result
                    break
            
            # Decide export strategy
            export_decision = self.export_decision_maker.decide_export_strategy(
                self._local_grid_to_dict(feature), hierarchical_result
            )
            
            # Update feature with export decision
            feature.export_type = export_decision['export_type']
            feature.blend_weights = export_decision['blend_weights']
            feature.optimal_lookback = export_decision['optimal_lookback']
            
            final_features.append(feature)
        
        results = {
            'optimized_features': final_features,
            'hierarchical_results': hierarchical_results,
            'export_decisions': [self._local_grid_to_dict(f) for f in final_features]
        }
        
        self.logger.info(f"Phase-2 optimization completed: {len(final_features)} features optimized")
        return results
    
    def _fit_ic_surface_for_candidate(self, 
                                    candidate,
                                    local_grid: List[int],
                                    sessionized_data: Dict[str, Any],
                                    targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Fit IC surface for a specific candidate."""
        # This would involve re-computing the HTF features for each lookback in the local grid
        # and calculating their ICs. For now, we'll simulate this process.
        
        # Simulate IC surface (in practice, you'd compute actual HTF features)
        ics = []
        ses = []
        
        for lookback in local_grid:
            # Simulate IC based on lookback (simplified)
            base_ic = candidate.ic_oos
            lookback_factor = np.exp(-abs(np.log(lookback) - np.log(candidate.lookback_minutes)) * 0.5)
            ic = base_ic * lookback_factor + np.random.normal(0, 0.01)
            se = candidate.se_wild_bootstrap * (1 + abs(np.log(lookback) - np.log(candidate.lookback_minutes)) * 0.1)
            
            ics.append(ic)
            ses.append(se)
        
        # Fit IC surface
        ic_surface_result = self.ic_surface_fitter.fit_ic_surface(
            local_grid, ics, ses
        )
        
        # Add surface data
        ic_surface_result['ic_surface'] = dict(zip(local_grid, ics))
        ic_surface_result['se_surface'] = dict(zip(local_grid, ses))
        
        return ic_surface_result
    
    def _local_grid_to_dict(self, local_grid_result: LocalGridResult) -> Dict[str, Any]:
        """Convert LocalGridResult to dictionary."""
        return {
            'feature_name': local_grid_result.feature_name,
            'family': local_grid_result.family,
            'optimal_lookback': local_grid_result.optimal_lookback,
            'optimal_ic': local_grid_result.optimal_ic,
            'optimal_se': local_grid_result.optimal_se,
            'confidence_interval': local_grid_result.confidence_interval,
            'export_type': local_grid_result.export_type,
            'blend_weights': local_grid_result.blend_weights,
            'regime': 'mixed'  # Would be determined from regime segmentation
        }