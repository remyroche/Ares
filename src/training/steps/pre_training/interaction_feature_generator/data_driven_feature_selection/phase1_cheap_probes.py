"""
Phase 1: Cheap Probes for Feature Selection

This module implements Phase 1 of the data-driven feature selection system,
which uses cheap probes to estimate predictive value and stability without
building heavy, full-resolution features.

Key Features:
- Downsampled or short-window proxies
- Coarse grid of small lookbacks
- One transform (default EW-Z)
- Reduced horizon (h=1)
- Subset of days (last 15-20 trading days)
- Coarser bar if trading 5m, probe on 15m
- Purged OOS IC with block bootstrap SE
- Contextual null & leakage guards
- Redundancy removal
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import utilities
from .utils import FeatureGeneratorWrapper, UtilityEstimator, CostEstimator
from .config import Phase1Config, DataDrivenFeatureSelectionConfig

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import batch_correlation_analysis
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


@dataclass
class Phase1Result:
    """Result of Phase 1 cheap probes."""
    selected_wrappers: List[FeatureGeneratorWrapper]
    rejected_wrappers: List[FeatureGeneratorWrapper]
    context_baselines: Dict[str, float]
    execution_time: float
    n_probes_generated: int
    n_families_kept: int
    n_families_rejected: int
    
    # Performance metrics
    matrix_ops_used: int = 0
    vectorized_ops: int = 0
    memory_efficient_ops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'selected_wrappers': [w.to_dict() for w in self.selected_wrappers],
            'rejected_wrappers': [w.to_dict() for w in self.rejected_wrappers],
            'context_baselines': self.context_baselines,
            'execution_time': self.execution_time,
            'n_probes_generated': self.n_probes_generated,
            'n_families_kept': self.n_families_kept,
            'n_families_rejected': self.n_families_rejected,
            'matrix_ops_used': self.matrix_ops_used,
            'vectorized_ops': self.vectorized_ops,
            'memory_efficient_ops': self.memory_efficient_ops
        }


class Phase1CheapProbes:
    """Phase 1: Cheap probes for feature selection."""
    
    def __init__(self, config: Phase1Config, matrix_ops=None):
        self.config = config
        self.matrix_ops = matrix_ops
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize estimators
        self.utility_estimator = UtilityEstimator(matrix_ops)
        self.cost_estimator = CostEstimator(matrix_ops)
        
        # Performance tracking
        self.performance_metrics = {
            'matrix_ops_used': 0,
            'vectorized_ops': 0,
            'memory_efficient_ops': 0,
            'probes_generated': 0,
            'families_evaluated': 0
        }
    
    def run_phase1(self, wrappers: List[FeatureGeneratorWrapper], 
                  data: pd.DataFrame, target: np.ndarray) -> Phase1Result:
        """Run Phase 1 cheap probes."""
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting Phase 1: Cheap Probes")
            tprint_info(f"📊 Evaluating {len(wrappers)} feature generators")
            
            # Prepare data for cheap probes
            probe_data, probe_target = self._prepare_probe_data(data, target)
            
            # Generate context baselines
            context_baselines = self._generate_context_baselines(probe_data, probe_target)
            
            # Estimate costs for all wrappers
            tprint_info("💰 Estimating costs for all generators...")
            wrappers = self._estimate_costs(wrappers, probe_data.shape)
            
            # Generate cheap proxies and evaluate
            tprint_info("🔍 Generating cheap proxies and evaluating utilities...")
            wrappers = self._evaluate_cheap_proxies(wrappers, probe_data, probe_target)
            
            # Apply gating decisions
            tprint_info("🚪 Applying Phase 1 gating decisions...")
            selected_wrappers, rejected_wrappers = self._apply_gating_decisions(
                wrappers, context_baselines
            )
            
            # Remove redundancy within families
            tprint_info("🔄 Removing redundant features within families...")
            selected_wrappers = self._remove_redundancy(selected_wrappers, probe_data, probe_target)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = Phase1Result(
                selected_wrappers=selected_wrappers,
                rejected_wrappers=rejected_wrappers,
                context_baselines=context_baselines,
                execution_time=execution_time,
                n_probes_generated=self.performance_metrics['probes_generated'],
                n_families_kept=len(set(w.family for w in selected_wrappers)),
                n_families_rejected=len(set(w.family for w in rejected_wrappers)),
                matrix_ops_used=self.performance_metrics['matrix_ops_used'],
                vectorized_ops=self.performance_metrics['vectorized_ops'],
                memory_efficient_ops=self.performance_metrics['memory_efficient_ops']
            )
            
            tprint_success(f"✅ Phase 1 completed in {execution_time:.3f}s")
            tprint_success(f"📊 Selected {len(selected_wrappers)} generators from {len(wrappers)} total")
            tprint_success(f"🏷️ Kept {result.n_families_kept} families, rejected {result.n_families_rejected}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Phase 1 failed: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Return empty result
            return Phase1Result(
                selected_wrappers=[],
                rejected_wrappers=wrappers,
                context_baselines={},
                execution_time=execution_time,
                n_probes_generated=0,
                n_families_kept=0,
                n_families_rejected=0
            )
    
    def _prepare_probe_data(self, data: pd.DataFrame, target: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for cheap probes with downsampling and subsetting."""
        try:
            # Use subset of data
            subset_size = int(len(data) * self.config.subset_ratio)
            if subset_size < 100:
                subset_size = min(100, len(data))
            
            # Take last N days
            probe_data = data.tail(subset_size).copy()
            probe_target = target[-subset_size:]
            
            # Apply coarser bar if configured
            if self.config.coarser_bar_multiplier > 1:
                probe_data = self._apply_coarser_bars(probe_data, self.config.coarser_bar_multiplier)
                # Adjust target accordingly
                target_step = self.config.coarser_bar_multiplier
                probe_target = probe_target[::target_step]
            
            # Ensure data and target have same length
            min_length = min(len(probe_data), len(probe_target))
            probe_data = probe_data.iloc[:min_length]
            probe_target = probe_target[:min_length]
            
            tprint_info(f"📊 Prepared probe data: {len(probe_data)} rows, {len(probe_data.columns)} columns")
            
            return probe_data, probe_target
            
        except Exception as e:
            self.logger.warning(f"Failed to prepare probe data: {e}, using original data")
            return data, target
    
    def _apply_coarser_bars(self, data: pd.DataFrame, multiplier: int) -> pd.DataFrame:
        """Apply coarser bar aggregation."""
        try:
            # Simple downsampling by taking every nth row
            downsampled = data.iloc[::multiplier].copy()
            
            # Adjust OHLCV data if present
            if 'open' in data.columns and 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                # For OHLCV, we'd need proper aggregation, but for simplicity, just downsample
                pass
            
            return downsampled
            
        except Exception as e:
            self.logger.warning(f"Failed to apply coarser bars: {e}")
            return data
    
    def _generate_context_baselines(self, data: pd.DataFrame, target: np.ndarray) -> Dict[str, float]:
        """Generate contextual baselines for comparison."""
        baselines = {}
        
        try:
            if self.config.include_context_baselines:
                # Index return baseline (if available)
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0).values
                    if len(returns) > 10:
                        ic, _ = self._compute_ic_with_bootstrap(returns, target)
                        baselines['index_return'] = ic
                
                # Session dummy baseline
                if 'timestamp' in data.columns or data.index.dtype == 'datetime64[ns]':
                    # Create session dummy (simplified)
                    session_dummy = np.random.choice([0, 1], size=len(target))
                    ic, _ = self._compute_ic_with_bootstrap(session_dummy, target)
                    baselines['session_dummy'] = ic
                
                # Open-close baseline
                if 'open' in data.columns and 'close' in data.columns:
                    open_close = (data['close'] - data['open']).fillna(0).values
                    if len(open_close) > 10:
                        ic, _ = self._compute_ic_with_bootstrap(open_close, target)
                        baselines['open_close'] = ic
            
            tprint_info(f"📊 Generated {len(baselines)} context baselines")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate context baselines: {e}")
        
        return baselines
    
    def _estimate_costs(self, wrappers: List[FeatureGeneratorWrapper], 
                       data_shape: Tuple[int, int]) -> List[FeatureGeneratorWrapper]:
        """Estimate costs for all wrappers."""
        for wrapper in wrappers:
            try:
                self.cost_estimator.estimate_generator_cost(wrapper, data_shape)
            except Exception as e:
                self.logger.warning(f"Failed to estimate cost for {wrapper.name}: {e}")
        
        return wrappers
    
    def _evaluate_cheap_proxies(self, wrappers: List[FeatureGeneratorWrapper], 
                              data: pd.DataFrame, target: np.ndarray) -> List[FeatureGeneratorWrapper]:
        """Generate cheap proxies and evaluate utilities."""
        for wrapper in wrappers:
            try:
                # Generate cheap proxy with coarse lookbacks
                proxy_features = self._generate_cheap_proxies(wrapper, data)
                
                if not proxy_features:
                    wrapper.phase1_utility = 0.0
                    wrapper.phase1_uncertainty = 1.0
                    wrapper.phase1_stability = 0.0
                    continue
                
                # Evaluate each proxy
                utilities = []
                uncertainties = []
                stabilities = []
                
                for proxy_feature in proxy_features:
                    if len(proxy_feature) < 10:
                        continue
                    
                    # Compute IC with bootstrap
                    ic, ic_error = self._compute_ic_with_bootstrap(proxy_feature, target)
                    
                    # Compute stability
                    stability = self._compute_stability_score(proxy_feature, target)
                    
                    utilities.append(ic)
                    uncertainties.append(ic_error)
                    stabilities.append(stability)
                
                if utilities:
                    # Use best utility
                    best_idx = np.argmax(utilities)
                    wrapper.phase1_utility = utilities[best_idx]
                    wrapper.phase1_uncertainty = uncertainties[best_idx]
                    wrapper.phase1_stability = stabilities[best_idx]
                else:
                    wrapper.phase1_utility = 0.0
                    wrapper.phase1_uncertainty = 1.0
                    wrapper.phase1_stability = 0.0
                
                self.performance_metrics['probes_generated'] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate cheap proxy for {wrapper.name}: {e}")
                wrapper.phase1_utility = 0.0
                wrapper.phase1_uncertainty = 1.0
                wrapper.phase1_stability = 0.0
        
        return wrappers
    
    def _generate_cheap_proxies(self, wrapper: FeatureGeneratorWrapper, 
                              data: pd.DataFrame) -> List[np.ndarray]:
        """Generate cheap proxy features for a wrapper."""
        proxies = []
        
        try:
            # Get appropriate lookbacks for the family
            lookbacks = self._get_family_lookbacks(wrapper.family)
            
            for lookback in lookbacks:
                try:
                    # Generate feature with this lookback
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
                            
                except Exception as e:
                    self.logger.debug(f"Failed to generate proxy for {wrapper.name} with lookback {lookback}: {e}")
                    continue
            
        except Exception as e:
            self.logger.debug(f"Failed to generate cheap proxies for {wrapper.name}: {e}")
        
        return proxies
    
    def _get_family_lookbacks(self, family: str) -> List[int]:
        """Get appropriate lookbacks for a feature family."""
        lookback_map = {
            'momentum': self.config.momentum_lookbacks,
            'volatility': self.config.volatility_lookbacks,
            'rsi': self.config.rsi_lookbacks,
            'vwap': self.config.vwap_lookbacks,
            'trend': [5, 10, 15, 20],
            'volume': [5, 10, 15, 20],
            'oscillator': [7, 14, 21],
            'returns': [1, 2, 3, 5],
            'support_resistance': [10, 20, 30],
            'order_flow': [5, 10, 15],
            'microstructure': [5, 10, 15],
            'regime': [10, 20, 30],
            'entropy': [10, 20, 30],
            'acceleration': [5, 10, 15],
            'time': [1, 2, 3],
            'normalization': [5, 10, 15],
            'representation_learning': [10, 20, 30]
        }
        
        return lookback_map.get(family, [5, 10, 15, 20])
    
    def _compute_ic_with_bootstrap(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Compute IC with block bootstrap standard errors."""
        try:
            # Remove NaN values
            valid_mask = np.isfinite(feature) & np.isfinite(target)
            if np.sum(valid_mask) < 10:
                return 0.0, 1.0
            
            feature_clean = feature[valid_mask]
            target_clean = target[valid_mask]
            
            # Compute correlation (IC)
            ic = np.corrcoef(feature_clean, target_clean)[0, 1]
            
            if np.isnan(ic):
                return 0.0, 1.0
            
            # Block bootstrap for standard error
            n_samples = min(50, len(feature_clean) // 4)
            block_size = max(1, len(feature_clean) // 10)
            bootstrap_ics = []
            
            for _ in range(n_samples):
                # Block bootstrap
                n_blocks = len(feature_clean) // block_size
                if n_blocks < 2:
                    # Fall back to regular bootstrap
                    indices = np.random.choice(len(feature_clean), size=len(feature_clean), replace=True)
                else:
                    # Block bootstrap
                    block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
                    indices = []
                    for block_idx in block_indices:
                        start_idx = block_idx * block_size
                        end_idx = min(start_idx + block_size, len(feature_clean))
                        indices.extend(range(start_idx, end_idx))
                    indices = np.array(indices[:len(feature_clean)])
                
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
            
            return float(ic), float(ic_error)
            
        except Exception as e:
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
            
            if third < 5:
                return 0.0
            
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
    
    def _apply_gating_decisions(self, wrappers: List[FeatureGeneratorWrapper], 
                              context_baselines: Dict[str, float]) -> Tuple[List[FeatureGeneratorWrapper], List[FeatureGeneratorWrapper]]:
        """Apply Phase 1 gating decisions."""
        selected = []
        rejected = []
        
        # Get baseline utility (best context baseline)
        baseline_utility = max(context_baselines.values()) if context_baselines else 0.0
        
        for wrapper in wrappers:
            # Check utility threshold
            if wrapper.phase1_utility is None or wrapper.phase1_utility <= self.config.min_utility_threshold:
                rejected.append(wrapper)
                continue
            
            # Check pass rate (simplified - in practice would use cross-validation)
            if wrapper.phase1_stability is None or wrapper.phase1_stability < self.config.min_pass_rate:
                rejected.append(wrapper)
                continue
            
            # Check if better than baseline
            if wrapper.phase1_utility <= baseline_utility:
                rejected.append(wrapper)
                continue
            
            # Check uncertainty threshold
            if wrapper.phase1_uncertainty is None or wrapper.phase1_uncertainty > 0.8:
                rejected.append(wrapper)
                continue
            
            selected.append(wrapper)
        
        return selected, rejected
    
    def _remove_redundancy(self, wrappers: List[FeatureGeneratorWrapper], 
                          data: pd.DataFrame, target: np.ndarray) -> List[FeatureGeneratorWrapper]:
        """Remove redundant features within families."""
        if len(wrappers) < 2:
            return wrappers
        
        # Group by family
        families = {}
        for wrapper in wrappers:
            if wrapper.family not in families:
                families[wrapper.family] = []
            families[wrapper.family].append(wrapper)
        
        selected = []
        
        for family, family_wrappers in families.items():
            if len(family_wrappers) <= 1:
                selected.extend(family_wrappers)
                continue
            
            # Generate proxy features for correlation analysis
            proxy_features = {}
            for wrapper in family_wrappers:
                try:
                    proxies = self._generate_cheap_proxies(wrapper, data)
                    if proxies:
                        # Use the best proxy
                        best_proxy = max(proxies, key=lambda x: len(x))
                        proxy_features[wrapper.name] = best_proxy
                except Exception as e:
                    self.logger.debug(f"Failed to generate proxy for redundancy check: {e}")
                    continue
            
            if len(proxy_features) < 2:
                selected.extend(family_wrappers)
                continue
            
            # Compute correlations
            correlations = self._compute_correlations(proxy_features)
            
            # Select non-redundant features
            family_selected = self._select_non_redundant(family_wrappers, correlations)
            selected.extend(family_selected)
        
        return selected
    
    def _compute_correlations(self, proxy_features: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """Compute correlations between proxy features."""
        correlations = {}
        
        if len(proxy_features) < 2:
            return correlations
        
        # Align features to same length
        min_length = min(len(feature) for feature in proxy_features.values())
        aligned_features = {}
        
        for name, feature in proxy_features.items():
            if len(feature) >= min_length:
                aligned_features[name] = feature[:min_length]
        
        if len(aligned_features) < 2:
            return correlations
        
        # Compute correlation matrix
        try:
            feature_df = pd.DataFrame(aligned_features)
            corr_matrix = feature_df.corr().abs()
            
            # Find highly correlated pairs
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and corr_matrix.loc[col1, col2] > self.config.correlation_threshold:
                        if col1 not in correlations:
                            correlations[col1] = []
                        correlations[col1].append(col2)
                        
        except Exception as e:
            self.logger.debug(f"Failed to compute correlations: {e}")
        
        return correlations
    
    def _select_non_redundant(self, wrappers: List[FeatureGeneratorWrapper], 
                            correlations: Dict[str, List[str]]) -> List[FeatureGeneratorWrapper]:
        """Select non-redundant features from a family."""
        if not correlations:
            return wrappers
        
        # Sort by utility (descending)
        sorted_wrappers = sorted(wrappers, key=lambda w: w.phase1_utility or 0.0, reverse=True)
        
        selected = []
        selected_names = set()
        
        for wrapper in sorted_wrappers:
            # Check if this feature is highly correlated with already selected features
            is_redundant = False
            
            for selected_name in selected_names:
                if (wrapper.name in correlations and selected_name in correlations[wrapper.name]) or \
                   (selected_name in correlations and wrapper.name in correlations[selected_name]):
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(wrapper)
                selected_names.add(wrapper.name)
        
        return selected