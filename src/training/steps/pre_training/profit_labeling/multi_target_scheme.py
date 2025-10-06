"""
Multi-Target Scheme for Volatility-Aware Labeling

This module implements the multi-target scheme (small/medium/high) with data-driven
selection of optimal parameters and horizons.

Key Features:
- Data-driven target selection within small/medium/high bands
- First-passage time (FPT) based horizon calculation
- Volatility-normalized target bands
- Quality-based target selection and filtering
- Mutual information assessment for target orthogonality
- Integration with Bayesian optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from scipy.stats import spearmanr
from scipy.optimize import minimize
import warnings

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZER_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available, using grid search")


class TargetBand(Enum):
    """Enumeration of target bands."""
    SMALL = "small"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MultiTargetConfig:
    """Configuration for multi-target scheme."""
    
    # Target band definitions
    small_band: Tuple[float, float] = (0.4, 0.8)  # k_s range
    medium_band: Tuple[float, float] = (0.8, 1.3)  # k_m range
    high_band: Tuple[float, float] = (1.3, 2.0)  # k_h range
    
    # Asymmetry options
    enable_asymmetry: bool = True
    asymmetry_ratios: List[float] = field(default_factory=lambda: [1.0, 1.25])
    
    # FPT (First-Passage Time) settings
    fpt_quantiles: List[float] = field(default_factory=lambda: [0.5, 0.65, 0.8])
    fpt_window: int = 100  # Window for FPT calculation
    fpt_min_samples: int = 50  # Minimum samples for FPT calculation
    
    # Horizon settings
    min_horizon: int = 1  # Minimum horizon in bars
    max_horizon: int = 100  # Maximum horizon in bars
    horizon_smoothing: bool = True
    horizon_ema_alpha: float = 0.1  # EMA alpha for horizon smoothing
    
    # Target selection
    max_targets_per_band: int = 2  # Maximum targets per band
    min_targets_total: int = 2  # Minimum total targets
    max_targets_total: int = 6  # Maximum total targets
    
    # Quality thresholds
    min_lqs_score: float = 0.3  # Minimum LQS score for target selection
    max_correlation_threshold: float = 0.6  # Maximum correlation between targets
    min_class_balance: float = 0.35  # Minimum class balance
    max_class_balance: float = 0.65  # Maximum class balance
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_method: str = 'bayesian'  # 'bayesian' or 'grid'
    n_trials: int = 100
    optimization_metric: str = 'lqs'  # 'lqs' or 'diversity'
    
    # Quality checks
    min_samples_per_target: int = 100
    max_evaluation_time_seconds: int = 300


@dataclass
class TargetSelectionResult:
    """Result container for target selection."""
    
    # Core results
    labels: pd.DataFrame
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    
    # Target information
    selected_targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_bands: Dict[str, TargetBand] = field(default_factory=dict)
    target_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Quality metrics
    target_quality_scores: Dict[str, float] = field(default_factory=dict)
    target_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    diversity_score: float = 0.0
    
    # Statistics
    n_targets: int = 0
    n_samples: int = 0
    target_coverage: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config_used: MultiTargetConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MultiTargetScheme:
    """
    Multi-Target Scheme for Volatility-Aware Labeling
    
    This class implements the multi-target scheme (small/medium/high) with data-driven
    selection of optimal parameters and horizons.
    
    Key Features:
    1. **Data-Driven Target Selection**: Searches within bands to find optimal k values
    2. **FPT-Based Horizons**: Uses first-passage time for adaptive horizon calculation
    3. **Volatility-Normalized Bands**: All targets are in σ-units
    4. **Quality-Based Selection**: Filters targets based on LQS scores
    5. **Orthogonality Assessment**: Ensures targets provide complementary signals
    6. **Bayesian Optimization**: Uses TPE for efficient parameter search
    """
    
    def __init__(self, config: Optional[MultiTargetConfig] = None):
        """Initialize multi-target scheme."""
        self.config = config or MultiTargetConfig()
        self.logger = logging.getLogger('MultiTargetScheme')
        
        tprint_info("🎯 Multi-Target Scheme initialized")
        tprint_info(f"   → Small band: {self.config.small_band}")
        tprint_info(f"   → Medium band: {self.config.medium_band}")
        tprint_info(f"   → High band: {self.config.high_band}")
        tprint_info(f"   → Optimization: {self.config.optimization_method}")
    
    def generate_targets(self, bars: pd.DataFrame, volatility_series: pd.Series,
                        eligibility_mask: pd.Series) -> TargetSelectionResult:
        """
        Generate multi-target labels with data-driven selection.
        
        Args:
            bars: Cleaned OHLCV bars
            volatility_series: Volatility estimates
            eligibility_mask: Eligibility mask from noise gating
            
        Returns:
            TargetSelectionResult with selected targets and labels
        """
        start_time = datetime.now()
        tprint_info("🎯 Generating multi-target labels")
        
        # Initialize result container
        result = TargetSelectionResult(
            labels=pd.DataFrame(),
            confidence_scores=pd.DataFrame(),
            eligibility_masks=pd.DataFrame(),
            config_used=self.config
        )
        
        try:
            # Validate input data
            if not self._validate_input_data(bars, volatility_series, eligibility_mask):
                return result
            
            # Align data
            common_index = bars.index.intersection(volatility_series.index).intersection(eligibility_mask.index)
            if len(common_index) == 0:
                tprint_warning("⚠️ No common index between inputs")
                return result
            
            bars_aligned = bars.loc[common_index]
            vol_aligned = volatility_series.loc[common_index]
            elig_aligned = eligibility_mask.loc[common_index]
            
            result.n_samples = len(common_index)
            
            # Step 1: Generate candidate targets
            tprint_info("📊 Step 1: Generating candidate targets")
            candidate_targets = self._generate_candidate_targets(bars_aligned, vol_aligned, elig_aligned)
            
            if not candidate_targets:
                tprint_warning("⚠️ No candidate targets generated")
                return result
            
            # Step 2: Calculate FPT-based horizons
            tprint_info("⏱️ Step 2: Calculating FPT-based horizons")
            horizons = self._calculate_fpt_horizons(candidate_targets, bars_aligned, vol_aligned)
            
            # Step 3: Generate labels for all candidates
            tprint_info("🏷️ Step 3: Generating labels for candidates")
            candidate_labels = self._generate_candidate_labels(
                candidate_targets, horizons, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Step 4: Assess quality and select targets
            tprint_info("📊 Step 4: Assessing quality and selecting targets")
            selected_targets = self._select_optimal_targets(candidate_labels, candidate_targets)
            
            if not selected_targets:
                tprint_warning("⚠️ No targets passed quality selection")
                return result
            
            # Step 5: Generate final labels
            tprint_info("✅ Step 5: Generating final labels")
            final_result = self._generate_final_labels(
                selected_targets, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Update result
            result.labels = final_result['labels']
            result.confidence_scores = final_result['confidence_scores']
            result.eligibility_masks = final_result['eligibility_masks']
            result.selected_targets = selected_targets
            result.n_targets = len(selected_targets)
            
            # Calculate additional metrics
            result.target_correlations = self._calculate_target_correlations(result.labels)
            result.diversity_score = self._calculate_diversity_score(result.labels)
            result.target_coverage = self._calculate_target_coverage(result.labels)
            
        except Exception as e:
            tprint_error(f"❌ Multi-target generation failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Multi-target generation completed")
        tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
        tprint_info(f"   → Selected targets: {result.n_targets}")
        tprint_info(f"   → Diversity score: {result.diversity_score:.3f}")
        
        return result
    
    def _validate_input_data(self, bars: pd.DataFrame, volatility_series: pd.Series,
                           eligibility_mask: pd.Series) -> bool:
        """Validate input data."""
        try:
            # Check if DataFrames are empty
            if bars.empty or volatility_series.empty or eligibility_mask.empty:
                tprint_warning("⚠️ Input data is empty")
                return False
            
            # Check required columns for bars
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = set(required_columns) - set(bars.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check for non-finite values
            if (bars[required_columns].isnull().any().any() or 
                volatility_series.isnull().any() or 
                eligibility_mask.isnull().any()):
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if (not np.isfinite(bars[required_columns].values).all() or 
                not np.isfinite(volatility_series.values).all() or
                not np.isfinite(eligibility_mask.values).all()):
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _generate_candidate_targets(self, bars: pd.DataFrame, volatility_series: pd.Series,
                                  eligibility_mask: pd.Series) -> List[Dict[str, Any]]:
        """Generate candidate targets within each band."""
        try:
            candidates = []
            
            # Generate candidates for each band
            for band in [TargetBand.SMALL, TargetBand.MEDIUM, TargetBand.HIGH]:
                band_candidates = self._generate_band_candidates(band, bars, volatility_series, eligibility_mask)
                candidates.extend(band_candidates)
            
            tprint_info(f"   → Generated {len(candidates)} candidate targets")
            return candidates
            
        except Exception as e:
            tprint_error(f"❌ Error generating candidate targets: {e}")
            return []
    
    def _generate_band_candidates(self, band: TargetBand, bars: pd.DataFrame,
                                volatility_series: pd.Series, eligibility_mask: pd.Series) -> List[Dict[str, Any]]:
        """Generate candidates for a specific band."""
        try:
            candidates = []
            
            # Get band range
            if band == TargetBand.SMALL:
                k_range = self.config.small_band
            elif band == TargetBand.MEDIUM:
                k_range = self.config.medium_band
            else:  # HIGH
                k_range = self.config.high_band
            
            # Generate k values within the band
            if self.config.optimization_method == 'bayesian' and BAYESIAN_OPTIMIZER_AVAILABLE:
                k_values = self._bayesian_optimize_k_values(k_range, bars, volatility_series, eligibility_mask, band)
            else:
                k_values = self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)
            
            # Generate candidates for each k value
            for k in k_values:
                for asymmetry in self.config.asymmetry_ratios:
                    candidate = {
                        'band': band,
                        'k_up': k,
                        'k_down': k * asymmetry,
                        'target_name': f"{band.value}_k{k:.2f}_a{asymmetry:.2f}",
                        'parameters': {
                            'k_up': k,
                            'k_down': k * asymmetry,
                            'band': band.value
                        }
                    }
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating candidates for band {band.value}: {e}")
            return []
    
    def _bayesian_optimize_k_values(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                                  volatility_series: pd.Series, eligibility_mask: pd.Series,
                                  band: TargetBand) -> List[float]:
        """Use Bayesian optimization to find optimal k values."""
        try:
            if not BAYESIAN_OPTIMIZER_AVAILABLE:
                return self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)
            
            # Define objective function
            def objective(k):
                try:
                    # Generate labels for this k value
                    labels = self._generate_labels_for_k(k, k, bars, volatility_series, eligibility_mask)
                    
                    if labels.empty:
                        return 0.0
                    
                    # Calculate quality score
                    quality_score = self._calculate_target_quality_score(labels, bars, volatility_series)
                    
                    return quality_score
                except Exception:
                    return 0.0
            
            # Set up optimization
            optimizer = BayesianTPEOptimizer(
                n_trials=self.config.n_trials,
                random_state=42
            )
            
            # Define search space
            search_space = {
                'k': (k_range[0], k_range[1])
            }
            
            # Run optimization
            best_params = optimizer.optimize(objective, search_space)
            
            # Extract k values
            k_values = [best_params['k']]
            
            # Add some additional k values around the optimal
            k_step = (k_range[1] - k_range[0]) / 10
            for offset in [-k_step, k_step]:
                k_val = best_params['k'] + offset
                if k_range[0] <= k_val <= k_range[1]:
                    k_values.append(k_val)
            
            return k_values
            
        except Exception as e:
            tprint_warning(f"⚠️ Bayesian optimization failed for band {band.value}: {e}")
            return self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)
    
    def _grid_search_k_values(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                            volatility_series: pd.Series, eligibility_mask: pd.Series,
                            band: TargetBand) -> List[float]:
        """Use grid search to find k values."""
        try:
            # Generate grid of k values
            n_points = min(10, self.config.n_trials)
            k_values = np.linspace(k_range[0], k_range[1], n_points)
            
            # Evaluate each k value
            k_scores = []
            for k in k_values:
                try:
                    labels = self._generate_labels_for_k(k, k, bars, volatility_series, eligibility_mask)
                    if not labels.empty:
                        quality_score = self._calculate_target_quality_score(labels, bars, volatility_series)
                        k_scores.append((k, quality_score))
                    else:
                        k_scores.append((k, 0.0))
                except Exception:
                    k_scores.append((k, 0.0))
            
            # Sort by quality score and return top k values
            k_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_values = [k for k, score in k_scores[:3] if score > 0]
            
            return top_k_values if top_k_values else [k_range[0] + (k_range[1] - k_range[0]) / 2]
            
        except Exception as e:
            tprint_warning(f"⚠️ Grid search failed for band {band.value}: {e}")
            return [k_range[0] + (k_range[1] - k_range[0]) / 2]
    
    def _generate_labels_for_k(self, k_up: float, k_down: float, bars: pd.DataFrame,
                             volatility_series: pd.Series, eligibility_mask: pd.Series) -> pd.Series:
        """Generate labels for specific k values."""
        try:
            # Calculate target levels
            upper_targets = bars['close'] + k_up * volatility_series
            lower_targets = bars['close'] - k_down * volatility_series
            
            # Initialize labels
            labels = pd.Series(0, index=bars.index)
            
            # Generate labels using triple barrier method
            for i in range(len(bars)):
                if not eligibility_mask.iloc[i]:
                    continue
                
                current_price = bars['close'].iloc[i]
                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]
                
                # Check if price hits targets in future
                future_prices = bars['close'].iloc[i+1:i+self.config.max_horizon]
                if len(future_prices) == 0:
                    continue
                
                # Find first hit
                upper_hits = future_prices >= upper_target
                lower_hits = future_prices <= lower_target
                
                if upper_hits.any() and lower_hits.any():
                    # Both hit - check which comes first
                    upper_first_hit = upper_hits.idxmax() if upper_hits.any() else None
                    lower_first_hit = lower_hits.idxmax() if lower_hits.any() else None
                    
                    if upper_first_hit is not None and lower_first_hit is not None:
                        if upper_first_hit <= lower_first_hit:
                            labels.iloc[i] = 1  # Upper hit first
                        else:
                            labels.iloc[i] = -1  # Lower hit first
                    elif upper_first_hit is not None:
                        labels.iloc[i] = 1
                    elif lower_first_hit is not None:
                        labels.iloc[i] = -1
                elif upper_hits.any():
                    labels.iloc[i] = 1
                elif lower_hits.any():
                    labels.iloc[i] = -1
            
            return labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels for k_up={k_up}, k_down={k_down}: {e}")
            return pd.Series(dtype=int, index=bars.index)
    
    def _calculate_target_quality_score(self, labels: pd.Series, bars: pd.DataFrame,
                                      volatility_series: pd.Series) -> float:
        """Calculate quality score for a target."""
        try:
            if labels.empty or labels.nunique() < 2:
                return 0.0
            
            # Calculate basic quality metrics
            class_balance = labels.value_counts().max() / len(labels)
            
            # Check if balance is within acceptable range
            if not (self.config.min_class_balance <= class_balance <= self.config.max_class_balance):
                return 0.0
            
            # Calculate information coefficient with volatility
            if len(volatility_series) > 0:
                common_index = labels.index.intersection(volatility_series.index)
                if len(common_index) > 10:
                    labels_aligned = labels.loc[common_index]
                    vol_aligned = volatility_series.loc[common_index]
                    
                    try:
                        ic, _ = spearmanr(labels_aligned, vol_aligned)
                        ic_score = abs(ic) if not np.isnan(ic) else 0.0
                    except Exception:
                        ic_score = 0.0
                else:
                    ic_score = 0.0
            else:
                ic_score = 0.0
            
            # Calculate flip rate
            flip_rate = (labels != labels.shift(1)).sum() / (len(labels) - 1) if len(labels) > 1 else 0.0
            
            # Combine metrics
            quality_score = (
                0.4 * (1.0 - abs(class_balance - 0.5) * 2) +  # Balance score
                0.3 * ic_score +  # Information coefficient
                0.3 * (1.0 - flip_rate)  # Stability score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating quality score: {e}")
            return 0.0
    
    def _calculate_fpt_horizons(self, candidate_targets: List[Dict[str, Any]],
                              bars: pd.DataFrame, volatility_series: pd.Series) -> Dict[str, int]:
        """Calculate first-passage time based horizons."""
        try:
            horizons = {}
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                
                # Calculate FPT for this target
                fpt = self._calculate_fpt_for_target(k_up, k_down, bars, volatility_series)
                
                if fpt is not None:
                    # Use quantile of FPT distribution as horizon
                    horizon = int(np.percentile(fpt, self.config.fpt_quantiles[1] * 100))  # Use middle quantile
                    horizon = max(self.config.min_horizon, min(self.config.max_horizon, horizon))
                    horizons[target_name] = horizon
                else:
                    horizons[target_name] = self.config.min_horizon
            
            return horizons
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT horizons: {e}")
            return {target['target_name']: self.config.min_horizon for target in candidate_targets}
    
    def _calculate_fpt_for_target(self, k_up: float, k_down: float, bars: pd.DataFrame,
                                volatility_series: pd.Series) -> Optional[np.ndarray]:
        """Calculate first-passage time for a specific target."""
        try:
            if len(bars) < self.config.fpt_min_samples:
                return None
            
            fpt_values = []
            
            for i in range(len(bars) - self.config.fpt_window):
                current_price = bars['close'].iloc[i]
                current_vol = volatility_series.iloc[i]
                
                if np.isnan(current_vol) or current_vol <= 0:
                    continue
                
                upper_target = current_price + k_up * current_vol
                lower_target = current_price - k_down * current_vol
                
                # Look ahead for first hit
                future_prices = bars['close'].iloc[i+1:i+self.config.fpt_window]
                
                for j, future_price in enumerate(future_prices):
                    if future_price >= upper_target or future_price <= lower_target:
                        fpt_values.append(j + 1)  # +1 because j is 0-indexed
                        break
            
            return np.array(fpt_values) if fpt_values else None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT for target: {e}")
            return None
    
    def _generate_candidate_labels(self, candidate_targets: List[Dict[str, Any]],
                                 horizons: Dict[str, int], bars: pd.DataFrame,
                                 volatility_series: pd.Series, eligibility_mask: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate labels for all candidate targets."""
        try:
            candidate_labels = {}
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                horizon = horizons.get(target_name, self.config.min_horizon)
                
                # Generate labels with specific horizon
                labels = self._generate_labels_with_horizon(
                    k_up, k_down, horizon, bars, volatility_series, eligibility_mask
                )
                
                if not labels.empty:
                    candidate_labels[target_name] = labels
            
            return candidate_labels
            
        except Exception as e:
            tprint_error(f"❌ Error generating candidate labels: {e}")
            return {}
    
    def _generate_labels_with_horizon(self, k_up: float, k_down: float, horizon: int,
                                    bars: pd.DataFrame, volatility_series: pd.Series,
                                    eligibility_mask: pd.Series) -> pd.Series:
        """Generate labels with specific horizon."""
        try:
            # Calculate target levels
            upper_targets = bars['close'] + k_up * volatility_series
            lower_targets = bars['close'] - k_down * volatility_series
            
            # Initialize labels
            labels = pd.Series(0, index=bars.index)
            confidence_scores = pd.Series(0.0, index=bars.index)
            
            # Generate labels using triple barrier method with horizon
            for i in range(len(bars) - horizon):
                if not eligibility_mask.iloc[i]:
                    continue
                
                current_price = bars['close'].iloc[i]
                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]
                
                # Check if price hits targets within horizon
                future_prices = bars['close'].iloc[i+1:i+horizon+1]
                if len(future_prices) == 0:
                    continue
                
                # Find first hit
                upper_hits = future_prices >= upper_target
                lower_hits = future_prices <= lower_target
                
                if upper_hits.any() and lower_hits.any():
                    # Both hit - check which comes first
                    upper_first_hit = upper_hits.idxmax() if upper_hits.any() else None
                    lower_first_hit = lower_hits.idxmax() if lower_hits.any() else None
                    
                    if upper_first_hit is not None and lower_first_hit is not None:
                        if upper_first_hit <= lower_first_hit:
                            labels.iloc[i] = 1  # Upper hit first
                            # Calculate confidence based on distance to opposite barrier
                            distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                            confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                        else:
                            labels.iloc[i] = -1  # Lower hit first
                            distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                            confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
                    elif upper_first_hit is not None:
                        labels.iloc[i] = 1
                        distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                        confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                    elif lower_first_hit is not None:
                        labels.iloc[i] = -1
                        distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                        confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
                elif upper_hits.any():
                    labels.iloc[i] = 1
                    distance_to_opposite = abs(future_prices.loc[upper_hits.idxmax()] - lower_target)
                    confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_down * volatility_series.iloc[i]))
                elif lower_hits.any():
                    labels.iloc[i] = -1
                    distance_to_opposite = abs(future_prices.loc[lower_hits.idxmax()] - upper_target)
                    confidence_scores.iloc[i] = min(1.0, distance_to_opposite / (k_up * volatility_series.iloc[i]))
            
            # Create DataFrame with labels and confidence
            result_df = pd.DataFrame({
                'labels': labels,
                'confidence': confidence_scores
            }, index=bars.index)
            
            return result_df
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels with horizon: {e}")
            return pd.DataFrame()
    
    def _select_optimal_targets(self, candidate_labels: Dict[str, pd.DataFrame],
                              candidate_targets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Select optimal targets based on quality and diversity."""
        try:
            if not candidate_labels:
                return {}
            
            # Calculate quality scores for all candidates
            quality_scores = {}
            for target_name, labels_df in candidate_labels.items():
                if not labels_df.empty and 'labels' in labels_df.columns:
                    labels = labels_df['labels']
                    quality_score = self._calculate_target_quality_score(labels, pd.DataFrame(), pd.Series())
                    quality_scores[target_name] = quality_score
            
            # Filter by minimum quality threshold
            qualified_targets = {
                name: score for name, score in quality_scores.items()
                if score >= self.config.min_lqs_score
            }
            
            if not qualified_targets:
                tprint_warning("⚠️ No targets passed quality threshold")
                return {}
            
            # Select targets by band
            selected_targets = {}
            band_counts = {band: 0 for band in TargetBand}
            
            # Sort by quality score
            sorted_targets = sorted(qualified_targets.items(), key=lambda x: x[1], reverse=True)
            
            for target_name, quality_score in sorted_targets:
                # Find the candidate info
                candidate_info = next((c for c in candidate_targets if c['target_name'] == target_name), None)
                if not candidate_info:
                    continue
                
                band = candidate_info['band']
                
                # Check band limits
                if band_counts[band] >= self.config.max_targets_per_band:
                    continue
                
                # Check total limits
                if len(selected_targets) >= self.config.max_targets_total:
                    break
                
                # Check correlation with already selected targets
                if self._check_correlation_constraints(target_name, selected_targets, candidate_labels):
                    selected_targets[target_name] = {
                        **candidate_info,
                        'quality_score': quality_score
                    }
                    band_counts[band] += 1
            
            # Ensure minimum targets
            if len(selected_targets) < self.config.min_targets_total:
                tprint_warning(f"⚠️ Only {len(selected_targets)} targets selected, minimum is {self.config.min_targets_total}")
            
            return selected_targets
            
        except Exception as e:
            tprint_error(f"❌ Error selecting optimal targets: {e}")
            return {}
    
    def _check_correlation_constraints(self, target_name: str, selected_targets: Dict[str, Any],
                                     candidate_labels: Dict[str, pd.DataFrame]) -> bool:
        """Check if target meets correlation constraints."""
        try:
            if not selected_targets:
                return True
            
            # Get labels for current target
            current_labels = candidate_labels.get(target_name)
            if current_labels is None or current_labels.empty or 'labels' not in current_labels.columns:
                return False
            
            current_labels_series = current_labels['labels']
            
            # Check correlation with each selected target
            for selected_name, selected_info in selected_targets.items():
                selected_labels = candidate_labels.get(selected_name)
                if selected_labels is None or selected_labels.empty or 'labels' not in selected_labels.columns:
                    continue
                
                selected_labels_series = selected_labels['labels']
                
                # Align indices
                common_index = current_labels_series.index.intersection(selected_labels_series.index)
                if len(common_index) < 10:
                    continue
                
                current_aligned = current_labels_series.loc[common_index]
                selected_aligned = selected_labels_series.loc[common_index]
                
                # Calculate correlation
                try:
                    corr, _ = spearmanr(current_aligned, selected_aligned)
                    if not np.isnan(corr) and abs(corr) > self.config.max_correlation_threshold:
                        return False
                except Exception:
                    continue
            
            return True
            
        except Exception as e:
            tprint_warning(f"⚠️ Error checking correlation constraints: {e}")
            return True
    
    def _generate_final_labels(self, selected_targets: Dict[str, Any], bars: pd.DataFrame,
                             volatility_series: pd.Series, eligibility_mask: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate final labels for selected targets."""
        try:
            labels_df = pd.DataFrame(index=bars.index)
            confidence_df = pd.DataFrame(index=bars.index)
            eligibility_df = pd.DataFrame(index=bars.index)
            
            for target_name, target_info in selected_targets.items():
                k_up = target_info['k_up']
                k_down = target_info['k_down']
                horizon = target_info.get('horizon', self.config.min_horizon)
                
                # Generate labels
                target_result = self._generate_labels_with_horizon(
                    k_up, k_down, horizon, bars, volatility_series, eligibility_mask
                )
                
                if not target_result.empty:
                    labels_df[target_name] = target_result['labels']
                    confidence_df[f"{target_name}_confidence"] = target_result['confidence']
                    eligibility_df[f"{target_name}_eligibility"] = eligibility_mask
            
            return {
                'labels': labels_df,
                'confidence_scores': confidence_df,
                'eligibility_masks': eligibility_df
            }
            
        except Exception as e:
            tprint_error(f"❌ Error generating final labels: {e}")
            return {
                'labels': pd.DataFrame(),
                'confidence_scores': pd.DataFrame(),
                'eligibility_masks': pd.DataFrame()
            }
    
    def _calculate_target_correlations(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between targets."""
        try:
            if labels_df.empty:
                return pd.DataFrame()
            
            # Calculate Spearman correlations
            corr_matrix = labels_df.corr(method='spearman')
            
            return corr_matrix
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target correlations: {e}")
            return pd.DataFrame()
    
    def _calculate_diversity_score(self, labels_df: pd.DataFrame) -> float:
        """Calculate diversity score for targets."""
        try:
            if labels_df.empty or len(labels_df.columns) < 2:
                return 0.0
            
            # Calculate average absolute correlation
            corr_matrix = labels_df.corr(method='spearman')
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Calculate average absolute correlation
            avg_abs_corr = upper_triangle.abs().mean().mean()
            
            # Diversity score (lower correlation = higher diversity)
            diversity_score = 1.0 - avg_abs_corr
            
            return max(0.0, min(1.0, diversity_score))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating diversity score: {e}")
            return 0.0
    
    def _calculate_target_coverage(self, labels_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate coverage for each target."""
        try:
            coverage = {}
            
            for col in labels_df.columns:
                if col in labels_df.columns:
                    non_zero_labels = (labels_df[col] != 0).sum()
                    total_samples = len(labels_df)
                    coverage[col] = non_zero_labels / total_samples if total_samples > 0 else 0.0
            
            return coverage
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating target coverage: {e}")
            return {}


# Convenience functions
def create_multi_target_scheme(config: Optional[MultiTargetConfig] = None) -> MultiTargetScheme:
    """Create multi-target scheme with specified configuration."""
    return MultiTargetScheme(config)


def generate_multi_targets(bars: pd.DataFrame, volatility_series: pd.Series,
                          eligibility_mask: pd.Series,
                          config: Optional[MultiTargetConfig] = None) -> TargetSelectionResult:
    """Generate multi-targets with default configuration."""
    scheme = MultiTargetScheme(config)
    return scheme.generate_targets(bars, volatility_series, eligibility_mask)