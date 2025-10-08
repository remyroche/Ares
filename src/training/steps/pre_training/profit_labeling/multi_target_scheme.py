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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import matrix operations for vectorized computations
try:
    from src.utils.matrix_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

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
class BandHorizonRule:
    """Configuration for adaptive horizon adjustments within a target band."""

    min_bars: Optional[int] = None
    max_bars: Optional[int] = None
    volatility_thresholds: List[Dict[str, float]] = field(default_factory=list)
    regime_multipliers: Dict[str, float] = field(default_factory=dict)
    default_multiplier: float = 1.0


@dataclass
class ForwardReturnSmoothingConfig:
    """Configuration for forward return smoothing by horizon."""

    enabled: bool = True
    default_lambda: float = 0.2
    per_horizon_lambdas: Dict[Union[int, str], float] = field(default_factory=dict)
    lambda_bounds: Tuple[float, float] = (0.05, 2.0)

    def _validate_config(self) -> None:
        """Validate smoothing configuration values."""
        if self.default_lambda <= 0:
            raise ValueError("default_lambda must be positive")

        lower, upper = self.lambda_bounds
        if lower <= 0 or upper <= 0 or lower >= upper:
            raise ValueError("lambda_bounds must be positive with lower < upper")

        for horizon, value in self.per_horizon_lambdas.items():
            if value <= 0:
                raise ValueError(
                    f"Smoothing lambda for horizon '{horizon}' must be positive"
                )


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

    # Adaptive horizon modifiers per band
    band_horizon_rules: Dict[TargetBand, BandHorizonRule] = field(
        default_factory=lambda: {
            TargetBand.SMALL: BandHorizonRule(),
            TargetBand.MEDIUM: BandHorizonRule(),
            TargetBand.HIGH: BandHorizonRule(),
        }
    )
    
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

    # Parallel processing settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = use all available cores
    parallel_method: str = 'thread'  # 'thread' or 'process'

    # Forward return smoothing configuration
    forward_return_smoothing: ForwardReturnSmoothingConfig = field(default_factory=ForwardReturnSmoothingConfig)

    def _validate_config(self) -> None:
        """Validate multi-target configuration values."""
        if self.min_horizon < 1:
            raise ValueError("min_horizon must be at least 1")
        if self.max_horizon < self.min_horizon:
            raise ValueError("max_horizon must be greater than or equal to min_horizon")

        if self.forward_return_smoothing:
            self.forward_return_smoothing._validate_config()


@dataclass
class TargetSelectionResult:
    """Result container for target selection."""

    # Core results
    labels: pd.DataFrame
    confidence_scores: pd.DataFrame
    eligibility_masks: pd.DataFrame
    sigma_payoffs: pd.DataFrame = field(default_factory=pd.DataFrame)
    training_labels: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_payoffs: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Target information
    selected_targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_bands: Dict[str, TargetBand] = field(default_factory=dict)
    target_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    target_shifts: Dict[str, int] = field(default_factory=dict)
    smoothing_settings: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
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
    selection_metadata: Dict[str, Any] = field(default_factory=dict)


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

        # Initialize matrix operations for vectorized computations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            tprint_warning("   → Matrix operations: Not available, using fallback")

        tprint_info("🎯 Multi-Target Scheme initialized")
        tprint_info(f"   → Small band: {self.config.small_band}")
        tprint_info(f"   → Medium band: {self.config.medium_band}")
        tprint_info(f"   → High band: {self.config.high_band}")
        tprint_info(f"   → Optimization: {self.config.optimization_method}")
        tprint_info(f"   → Parallel processing: {self.config.enable_parallel_processing}")
    
    def generate_targets(self, bars: pd.DataFrame, volatility_series: pd.Series,
                        eligibility_mask: pd.Series,
                        regime_context: Optional[Dict[str, Any]] = None) -> TargetSelectionResult:
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
            sigma_payoffs=pd.DataFrame(),
            training_labels=pd.DataFrame(),
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
            horizons = self._calculate_fpt_horizons(
                candidate_targets, bars_aligned, vol_aligned, regime_context
            )

            for candidate in candidate_targets:
                horizon_info = horizons.get(candidate['target_name']) if isinstance(horizons, dict) else None
                if isinstance(horizon_info, dict):
                    candidate['horizon'] = horizon_info.get('horizon', self.config.min_horizon)
                    candidate['horizon_context'] = horizon_info
                else:
                    candidate['horizon'] = horizon_info or self.config.min_horizon
                    candidate['horizon_context'] = {'horizon': candidate['horizon']}
                candidate.setdefault('parameters', {})
                candidate['parameters']['horizon'] = candidate['horizon']
                candidate['target_shift'] = max(1, int(candidate.get('target_shift', 1)))
                candidate['parameters']['target_shift'] = candidate['target_shift']

            self._apply_smoothing_parameters(candidate_targets)

            # Step 3: Generate labels for all candidates
            tprint_info("🏷️ Step 3: Generating labels for candidates")
            candidate_labels = self._generate_candidate_labels(
                candidate_targets, horizons, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Step 4: Assess quality and select targets
            tprint_info("📊 Step 4: Assessing quality and selecting targets")
            selected_targets, selection_metadata = self._select_optimal_targets(
                candidate_labels, candidate_targets
            )
            result.selection_metadata = selection_metadata

            if not selected_targets:
                tprint_warning("⚠️ No targets passed quality selection")
                return result
            
            # Step 5: Generate final labels
            tprint_info("✅ Step 5: Generating final labels")
            final_result = self._generate_final_labels(
                selected_targets, bars_aligned, vol_aligned, elig_aligned
            )
            
            # Step 6: Apply label smoothing and conflict resolution
            tprint_info("🔧 Step 6: Applying label smoothing and conflict resolution")
            if not final_result['labels'].empty:
                # Resolve conflicts
                final_result['labels'] = self._resolve_label_conflicts(final_result['labels'])
                
                # Apply label smoothing
                final_result['labels'] = self._apply_label_smoothing(
                    final_result['labels'], 
                    final_result['confidence_scores']
                )
            
            # Update result
            result.labels = final_result['labels']
            result.confidence_scores = final_result['confidence_scores']
            result.eligibility_masks = final_result['eligibility_masks']
            result.sigma_payoffs = final_result.get('sigma_payoffs', pd.DataFrame())
            result.raw_payoffs = final_result.get('raw_payoffs', pd.DataFrame())
            result.training_labels = result.labels.copy()
            result.selected_targets = selected_targets
            result.n_targets = len(selected_targets)
            result.target_parameters = {
                name: {
                    **(info.get('parameters', {})),
                    'horizon': info.get('horizon'),
                    'horizon_context': info.get('horizon_context'),
                    'target_shift': info.get('target_shift', 1),
                }
                for name, info in selected_targets.items()
            }
            result.target_shifts = {
                name: int(info.get('target_shift', 1))
                for name, info in selected_targets.items()
            }
            result.smoothing_settings = self._extract_smoothing_settings(result.target_parameters)
            result.target_bands = {
                name: info.get('band') for name, info in selected_targets.items()
            }

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
        """Generate candidate targets within each band using parallel processing."""
        try:
            bands = [TargetBand.SMALL, TargetBand.MEDIUM, TargetBand.HIGH]

            # Create tasks for parallel execution
            def create_band_task(band):
                return lambda: self._generate_band_candidates(band, bars, volatility_series, eligibility_mask)

            tasks = [create_band_task(band) for band in bands]

            # Execute in parallel
            band_results = self._execute_parallel(tasks)

            # Combine results
            candidates = []
            for band_result in band_results:
                if band_result:
                    candidates.extend(band_result)

            tprint_info(f"   → Generated {len(candidates)} candidate targets")
            return candidates

        except Exception as e:
            tprint_error(f"❌ Error generating candidate targets: {e}")
            return []

    def _generate_band_candidates(self, band: TargetBand, bars: pd.DataFrame,
                                volatility_series: pd.Series, eligibility_mask: pd.Series) -> List[Dict[str, Any]]:
        """Generate candidates for a specific band with conditional thresholds."""
        try:
            candidates = []
            
            # Get band range
            if band == TargetBand.SMALL:
                k_range = self.config.small_band
            elif band == TargetBand.MEDIUM:
                k_range = self.config.medium_band
            else:  # HIGH
                k_range = self.config.high_band
                # Apply conditional thresholds for high targets based on volatility
                k_range = self._apply_conditional_thresholds(k_range, volatility_series, band)
            
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
                        'target_shift': 1,
                        'parameters': {
                            'k_up': k,
                            'k_down': k * asymmetry,
                            'band': band.value,
                            'target_shift': 1,
                        }
                    }
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating candidates for band {band.value}: {e}")
            return []

    def _apply_smoothing_parameters(self, candidate_targets: List[Dict[str, Any]]) -> None:
        """Attach decay lambda parameters to each candidate based on horizon."""
        try:
            smoothing_cfg = getattr(self.config, 'forward_return_smoothing', None)
            if not smoothing_cfg or not smoothing_cfg.enabled:
                return

            for candidate in candidate_targets:
                horizon_value = candidate.get('horizon', self.config.min_horizon)
                decay_lambda = self._determine_decay_lambda(horizon_value)
                candidate.setdefault('parameters', {})['decay_lambda'] = decay_lambda
        except Exception as e:
            tprint_warning(f"⚠️ Error applying smoothing parameters: {e}")

    def _determine_decay_lambda(self, horizon: Optional[Union[int, float]]) -> float:
        """Resolve decay lambda for a given horizon using configuration overrides."""
        smoothing_cfg = getattr(self.config, 'forward_return_smoothing', None)
        if not smoothing_cfg:
            return 0.0

        lambda_value: Optional[float] = None
        if horizon is not None and smoothing_cfg.per_horizon_lambdas:
            horizon_key_int = int(round(float(horizon)))
            if horizon_key_int in smoothing_cfg.per_horizon_lambdas:
                lambda_value = smoothing_cfg.per_horizon_lambdas[horizon_key_int]
            elif str(horizon_key_int) in smoothing_cfg.per_horizon_lambdas:
                lambda_value = smoothing_cfg.per_horizon_lambdas[str(horizon_key_int)]

        if lambda_value is None:
            lambda_value = smoothing_cfg.default_lambda

        lower, upper = smoothing_cfg.lambda_bounds
        if lower is not None:
            lambda_value = max(lower, lambda_value)
        if upper is not None:
            lambda_value = min(upper, lambda_value)

        return float(lambda_value)

    def _extract_smoothing_settings(self, target_parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Build smoothing metadata for selected targets."""
        smoothing_cfg = getattr(self.config, 'forward_return_smoothing', None)
        if not smoothing_cfg or not smoothing_cfg.enabled:
            return {}

        smoothing_settings: Dict[str, Dict[str, float]] = {}
        for name, params in (target_parameters or {}).items():
            if params is None:
                continue

            horizon_value = params.get('horizon')
            decay_lambda = params.get('decay_lambda')
            if decay_lambda is None:
                decay_lambda = self._determine_decay_lambda(horizon_value)
                params['decay_lambda'] = decay_lambda

            halflife = self._lambda_to_halflife(decay_lambda) if decay_lambda > 0 else 0.0
            smoothing_settings[name] = {
                'decay_lambda': float(decay_lambda),
                'halflife': float(halflife),
                'horizon': float(horizon_value) if horizon_value is not None else None,
                'method': 'ewm_halflife',
                'aggregation': 'exponential_weighted_mean'
            }

        return smoothing_settings

    @staticmethod
    def _lambda_to_halflife(decay_lambda: float) -> float:
        """Convert exponential decay lambda to half-life units."""
        return float(np.log(2) / max(decay_lambda, 1e-12))
    
    def _apply_conditional_thresholds(self, k_range: Tuple[float, float], 
                                    volatility_series: pd.Series, band: TargetBand) -> Tuple[float, float]:
        """Apply conditional thresholds for high targets based on volatility."""
        try:
            if band != TargetBand.HIGH:
                return k_range
            
            # Calculate volatility percentiles
            vol_25 = volatility_series.quantile(0.25)
            vol_75 = volatility_series.quantile(0.75)
            vol_median = volatility_series.median()
            
            # Adjust k range based on volatility
            if vol_median < vol_25:
                # Low volatility: use higher k values (2.0)
                adjusted_range = (max(k_range[0], 1.8), min(k_range[1], 2.2))
            elif vol_median > vol_75:
                # High volatility: use lower k values (1.5)
                adjusted_range = (max(k_range[0], 1.2), min(k_range[1], 1.8))
            else:
                # Medium volatility: use original range
                adjusted_range = k_range
            
            tprint_info(f"   📊 Adjusted {band.value} band range: {adjusted_range} (volatility: {vol_median:.4f})")
            
            return adjusted_range
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying conditional thresholds: {e}")
            return k_range
    
    def _bayesian_optimize_k_values(self, k_range: Tuple[float, float], bars: pd.DataFrame,
                                  volatility_series: pd.Series, eligibility_mask: pd.Series,
                                  band: TargetBand) -> List[float]:
        """Use adaptive sampling with early stopping for efficient O(log n) optimization."""
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

            # Adaptive sampling strategy for O(log n) complexity
            tprint_info(f"   🔍 Adaptive optimization for {band.value} band")

            # Step 1: Initial sparse sampling (logarithmic spacing)
            n_initial = min(8, int(np.log2(self.config.n_trials)) + 2)  # Start with log-scale samples
            initial_k_values = self._adaptive_initial_sampling(k_range, n_initial)

            # Evaluate initial samples in parallel
            def evaluate_k(k):
                return (k, objective(k))

            initial_tasks = [lambda k=k: evaluate_k(k) for k in initial_k_values]
            initial_results = self._execute_parallel(initial_tasks)
            initial_scores = [result for result in initial_results if result is not None]

            # Early stopping check - if we found good solutions, return early
            good_solutions = [(k, score) for k, score in initial_scores if score > 0.7]
            if len(good_solutions) >= 2:
                tprint_info(f"   ✅ Early stopping: Found {len(good_solutions)} good solutions")
                return [k for k, score in sorted(good_solutions, key=lambda x: x[1], reverse=True)[:3]]

            # Step 2: Adaptive refinement around best regions
            best_k = max(initial_scores, key=lambda x: x[1])[0]
            refined_k_values = self._adaptive_refinement(k_range, best_k, initial_scores, objective)

            # Combine and evaluate refinement samples in parallel
            if refined_k_values:
                refined_tasks = [lambda k=k: evaluate_k(k) for k in refined_k_values]
                refined_results = self._execute_parallel(refined_tasks)
                refined_scores = [result for result in refined_results if result is not None]
                all_scores = initial_scores + refined_scores
            else:
                all_scores = initial_scores

            # Sort by quality score and return top values
            all_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_values = [k for k, score in all_scores[:3] if score > 0]

            return top_k_values if top_k_values else [k_range[0] + (k_range[1] - k_range[0]) / 2]

        except Exception as e:
            tprint_warning(f"⚠️ Adaptive optimization failed for band {band.value}: {e}")
            return self._grid_search_k_values(k_range, bars, volatility_series, eligibility_mask, band)

    def _execute_parallel(self, tasks: List[callable], max_workers: Optional[int] = None) -> List[Any]:
        """Execute tasks in parallel using thread or process pool."""
        if not self.config.enable_parallel_processing or len(tasks) <= 1:
            # Fallback to sequential execution
            return [task() for task in tasks]

        try:
            max_workers = max_workers or self.config.max_workers or min(mp.cpu_count(), len(tasks))

            if self.config.parallel_method == 'process':
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(task): task for task in tasks}

                # Collect results as they complete
                results = []
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        tprint_warning(f"⚠️ Parallel task failed: {e}")
                        results.append(None)

                return results

        except Exception as e:
            tprint_warning(f"⚠️ Parallel execution failed: {e}")
            # Fallback to sequential
            return [task() for task in tasks]

    def _adaptive_initial_sampling(self, k_range: Tuple[float, float], n_points: int) -> List[float]:
        """Generate initial samples using logarithmic spacing for efficient exploration."""
        try:
            # Use logarithmic spacing to cover the range more efficiently
            log_min = np.log(k_range[0] + 1e-8)  # Avoid log(0)
            log_max = np.log(k_range[1] + 1e-8)
            log_samples = np.linspace(log_min, log_max, n_points)
            k_values = [np.exp(log_k) - 1e-8 for log_k in log_samples]

            # Ensure boundaries are included
            k_values[0] = k_range[0]
            k_values[-1] = k_range[1]

            return k_values
        except Exception:
            # Fallback to linear spacing
            return list(np.linspace(k_range[0], k_range[1], n_points))

    def _adaptive_refinement(self, k_range: Tuple[float, float], best_k: float,
                           initial_scores: List[Tuple[float, float]], objective: callable) -> List[float]:
        """Adaptive refinement around the best region found."""
        try:
            # Find the range around the best solution
            sorted_scores = sorted(initial_scores, key=lambda x: x[1], reverse=True)
            best_score = sorted_scores[0][1]

            # If we have a very good solution, do minimal refinement
            if best_score > 0.8:
                return []

            # Calculate adaptive range based on score quality
            score_range = max(0.1, 1.0 - best_score)  # Larger range for worse solutions
            refinement_range = score_range * (k_range[1] - k_range[0]) * 0.3

            # Refine around the best k value
            k_min = max(k_range[0], best_k - refinement_range)
            k_max = min(k_range[1], best_k + refinement_range)

            # Generate refinement points
            n_refine = min(6, int(np.log2(self.config.n_trials)) + 1)
            refined_values = np.linspace(k_min, k_max, n_refine)

            return list(refined_values)
        except Exception:
            return []

    def _coarse_grid_search(self, k_range: Tuple[float, float], objective: callable, n_points: int = 20) -> List[float]:
        """Coarse grid search to identify promising regions."""
        try:
            k_values = np.linspace(k_range[0], k_range[1], n_points)
            scores = []
            
            for k in k_values:
                score = objective(k)
                scores.append(score)
            
            # Find regions with high scores
            scores = np.array(scores)
            threshold = np.percentile(scores, 70)  # Top 30% of scores
            
            promising_k_values = k_values[scores >= threshold].tolist()
            
            return promising_k_values
            
        except Exception as e:
            tprint_warning(f"⚠️ Coarse grid search failed: {e}")
            return []
    
    def _fine_grid_search(self, promising_k_values: List[float], objective: callable, n_points: int = 15) -> List[float]:
        """Fine grid search around promising regions."""
        try:
            if not promising_k_values:
                return []
            
            # Create fine grid around promising values
            fine_k_values = []
            
            for k in promising_k_values:
                # Create local grid around this k value
                local_range = 0.1 * (max(promising_k_values) - min(promising_k_values))
                local_k_values = np.linspace(
                    max(k - local_range, min(promising_k_values)),
                    min(k + local_range, max(promising_k_values)),
                    n_points
                )
                fine_k_values.extend(local_k_values)
            
            # Remove duplicates and evaluate
            fine_k_values = list(set(fine_k_values))
            scores = [objective(k) for k in fine_k_values]
            
            # Return top values
            k_scores = list(zip(fine_k_values, scores))
            k_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [k for k, score in k_scores[:5] if score > 0]
            
        except Exception as e:
            tprint_warning(f"⚠️ Fine grid search failed: {e}")
            return []
    
    def _tpe_optimization(self, fine_k_values: List[float], objective: callable, k_range: Tuple[float, float]) -> List[float]:
        """TPE optimization in the best region."""
        try:
            if not fine_k_values or not BAYESIAN_OPTIMIZER_AVAILABLE:
                return []
            
            # Define search space around fine grid results
            k_min = min(fine_k_values)
            k_max = max(fine_k_values)
            
            # Expand range slightly for TPE
            range_expansion = 0.1 * (k_max - k_min)
            tpe_k_min = max(k_min - range_expansion, k_range[0])
            tpe_k_max = min(k_max + range_expansion, k_range[1])
            
            # Set up TPE optimizer
            optimizer = BayesianTPEOptimizer(
                n_trials=min(50, self.config.n_trials // 2),
                random_state=42
            )
            
            # Define search space
            search_space = {
                'k': (tpe_k_min, tpe_k_max)
            }
            
            # Run TPE optimization
            best_params = optimizer.optimize(objective, search_space)
            
            return [best_params['k']]
            
        except Exception as e:
            tprint_warning(f"⚠️ TPE optimization failed: {e}")
            return []
    
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
        """Generate labels for specific k values using vectorized operations."""
        try:
            n_bars = len(bars)
            max_horizon = self.config.max_horizon

            # Initialize labels
            labels = pd.Series(0, index=bars.index)

            # Vectorized target level calculation
            upper_targets = bars['close'] + k_up * volatility_series
            lower_targets = bars['close'] - k_down * volatility_series

            # Create rolling windows for future price comparison
            # This is more complex to vectorize fully, but we can optimize the inner loop

            for i in range(n_bars):
                if not eligibility_mask.iloc[i]:
                    continue

                # Get future prices for this bar
                future_prices = bars['close'].iloc[i+1:i+max_horizon+1]
                if len(future_prices) == 0:
                    continue

                upper_target = upper_targets.iloc[i]
                lower_target = lower_targets.iloc[i]

                # Vectorized hit detection for this bar's future prices
                upper_hits = future_prices >= upper_target
                lower_hits = future_prices <= lower_target

                # Use matrix operations for efficient first-hit detection if available
                if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                    # Use matrix operations for efficient argmax computation
                    upper_hit_indices = self._vectorized_first_hit(upper_hits.values)
                    lower_hit_indices = self._vectorized_first_hit(lower_hits.values)
                else:
                    # Fallback to numpy operations
                    upper_hit_indices = upper_hits.values.argmax() if upper_hits.any() else -1
                    lower_hit_indices = lower_hits.values.argmax() if lower_hits.any() else -1

                # Determine label based on first hits
                if upper_hit_indices >= 0 and lower_hit_indices >= 0:
                    if upper_hit_indices <= lower_hit_indices:
                        labels.iloc[i] = 1  # Upper hit first
                    else:
                        labels.iloc[i] = -1  # Lower hit first
                elif upper_hit_indices >= 0:
                    labels.iloc[i] = 1
                elif lower_hit_indices >= 0:
                    labels.iloc[i] = -1

            return labels

        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels for k_up={k_up}, k_down={k_down}: {e}")
            return pd.Series(dtype=int, index=bars.index)

    def _vectorized_first_hit(self, hit_array: np.ndarray) -> int:
        """Vectorized first hit detection using matrix operations."""
        try:
            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Use matrix operations for efficient first True detection
                if len(hit_array) == 0:
                    return -1

                # Create cumulative sum to find first occurrence
                cumsum = np.cumsum(hit_array.astype(int))
                first_hit_idx = np.where(cumsum == 1)[0]

                return first_hit_idx[0] if len(first_hit_idx) > 0 else -1
            else:
                # Fallback to numpy argmax
                return hit_array.argmax() if hit_array.any() else -1

        except Exception as e:
            tprint_warning(f"⚠️ Vectorized first hit detection failed: {e}")
            return hit_array.argmax() if hit_array.any() else -1
    
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
                              bars: pd.DataFrame, volatility_series: pd.Series,
                              regime_context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Calculate first-passage time based horizons with adaptive modifiers."""
        try:
            horizons: Dict[str, Dict[str, Any]] = {}

            active_regime = None
            regime_series = None
            regime_masks: Dict[str, pd.Series] = {}

            if regime_context:
                regime_series = regime_context.get('regime_series')
                if isinstance(regime_series, pd.Series) and not regime_series.empty:
                    active_regime = regime_context.get('active_regime', regime_series.iloc[-1])
                else:
                    active_regime = regime_context.get('active_regime')

                regime_masks = {
                    name: mask for name, mask in (regime_context.get('regime_masks') or {}).items()
                    if isinstance(mask, pd.Series)
                }

                if active_regime is None and regime_masks:
                    # Use mask coverage to infer dominant regime
                    coverage_scores = {
                        name: float(mask.reindex(bars.index, fill_value=False).mean())
                        for name, mask in regime_masks.items() if len(mask) > 0
                    }
                    if coverage_scores:
                        active_regime = max(coverage_scores.items(), key=lambda item: item[1])[0]

            vol_metric: Optional[float] = None
            if len(volatility_series) > 0:
                vol_array = pd.to_numeric(volatility_series, errors='coerce').values
                finite_mask = np.isfinite(vol_array)
                if finite_mask.any():
                    vol_metric = float(np.median(vol_array[finite_mask]))

            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']
                band = candidate.get('band', TargetBand.MEDIUM)

                # Calculate FPT for this target
                fpt = self._calculate_fpt_for_target(k_up, k_down, bars, volatility_series)

                if fpt is not None and len(fpt) > 0:
                    # Use middle quantile of FPT distribution as base horizon
                    base_horizon = max(self.config.min_horizon, int(fpt[1]))
                else:
                    base_horizon = self.config.min_horizon

                rule = self.config.band_horizon_rules.get(band, BandHorizonRule())
                multiplier = rule.default_multiplier or 1.0
                matched_vol_rule: Optional[Dict[str, float]] = None

                if vol_metric is not None and not np.isnan(vol_metric) and rule.volatility_thresholds:
                    for threshold in rule.volatility_thresholds:
                        min_vol = threshold.get('min', -np.inf)
                        max_vol = threshold.get('max', np.inf)
                        if min_vol <= vol_metric <= max_vol:
                            multiplier *= threshold.get('multiplier', 1.0)
                            matched_vol_rule = threshold
                            break

                applied_regime = active_regime
                if rule.regime_multipliers and active_regime in rule.regime_multipliers:
                    multiplier *= rule.regime_multipliers[active_regime]
                elif rule.regime_multipliers and regime_masks:
                    # Check for regime masks with highest recent coverage
                    mask_scores = {
                        name: float(mask.reindex(bars.index, fill_value=False).tail(base_horizon).mean())
                        for name, mask in regime_masks.items() if len(mask) > 0
                    }
                    if mask_scores:
                        applied_regime, score = max(mask_scores.items(), key=lambda item: item[1])
                        if applied_regime in rule.regime_multipliers:
                            multiplier *= rule.regime_multipliers[applied_regime]

                adjusted_horizon = int(round(base_horizon * max(multiplier, 0.0)))

                # Clamp using band rules first, then global config
                band_min = rule.min_bars if rule.min_bars is not None else self.config.min_horizon
                band_max = rule.max_bars if rule.max_bars is not None else self.config.max_horizon

                adjusted_horizon = max(band_min, adjusted_horizon)
                adjusted_horizon = min(band_max, adjusted_horizon)
                adjusted_horizon = max(self.config.min_horizon, min(self.config.max_horizon, adjusted_horizon))

                horizons[target_name] = {
                    'horizon': adjusted_horizon,
                    'base_horizon': base_horizon,
                    'applied_multiplier': multiplier,
                    'volatility_metric': vol_metric,
                    'matched_volatility_rule': matched_vol_rule,
                    'active_regime': applied_regime,
                    'band_min': band_min,
                    'band_max': band_max,
                }

            return horizons

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT horizons: {e}")
            return {
                target['target_name']: {
                    'horizon': self.config.min_horizon,
                    'base_horizon': self.config.min_horizon,
                    'applied_multiplier': 1.0,
                    'volatility_metric': None,
                    'matched_volatility_rule': None,
                    'active_regime': None,
                    'band_min': self.config.min_horizon,
                    'band_max': self.config.max_horizon,
                }
                for target in candidate_targets
            }
    
    def _calculate_fpt_for_target(self, k_up: float, k_down: float, bars: pd.DataFrame,
                                volatility_series: pd.Series) -> Optional[np.ndarray]:
        """Calculate first-passage time for a specific target using survival analysis approach."""
        try:
            if len(bars) < self.config.fpt_min_samples:
                return None
            
            fpt_values = []
            censored_values = []  # For survival analysis
            
            for i in range(len(bars) - self.config.fpt_window):
                current_price = bars['close'].iloc[i]
                current_vol = volatility_series.iloc[i]
                
                if np.isnan(current_vol) or current_vol <= 0:
                    continue
                
                upper_target = current_price + k_up * current_vol
                lower_target = current_price - k_down * current_vol
                
                # Look ahead for first hit
                future_prices = bars['close'].iloc[i+1:i+self.config.fpt_window]
                
                hit_time = None
                for j, future_price in enumerate(future_prices):
                    if future_price >= upper_target or future_price <= lower_target:
                        hit_time = j + 1  # +1 because j is 0-indexed
                        break
                
                if hit_time is not None:
                    fpt_values.append(hit_time)
                else:
                    # Censored observation (no hit within window)
                    censored_values.append(self.config.fpt_window)
            
            # Use survival analysis approach for better FPT estimation
            if fpt_values:
                # Calculate Kaplan-Meier-like estimator for FPT distribution
                fpt_array = np.array(fpt_values)
                censored_array = np.array(censored_values) if censored_values else np.array([])
                
                # Combine observed and censored times
                all_times = np.concatenate([fpt_array, censored_array])
                event_indicators = np.concatenate([
                    np.ones(len(fpt_array)),  # 1 for observed events
                    np.zeros(len(censored_array))  # 0 for censored
                ])
                
                # Sort by time
                sort_idx = np.argsort(all_times)
                sorted_times = all_times[sort_idx]
                sorted_events = event_indicators[sort_idx]
                
                # Calculate survival probabilities
                survival_probs = self._calculate_survival_probabilities(sorted_times, sorted_events)
                
                # Use quantiles of survival distribution for FPT estimation
                quantile_times = []
                for q in self.config.fpt_quantiles:
                    # Find time where survival probability drops below (1-q)
                    target_survival = 1 - q
                    idx = np.where(survival_probs <= target_survival)[0]
                    if len(idx) > 0:
                        quantile_times.append(sorted_times[idx[0]])
                    else:
                        quantile_times.append(sorted_times[-1])
                
                return np.array(quantile_times)
            else:
                return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating FPT for target: {e}")
            return None
    
    def _calculate_survival_probabilities(self, times: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Calculate survival probabilities using Kaplan-Meier estimator."""
        try:
            n = len(times)
            survival_probs = np.ones(n)

            # Calculate at-risk counts (number of individuals at risk at each time point)
            at_risk = np.zeros(n, dtype=int)
            for i in range(n):
                at_risk[i] = n - i  # At time i, there are (n-i) individuals still at risk

            # Calculate survival probabilities using proper Kaplan-Meier formula
            for i in range(n):
                if events[i] == 1:  # Observed event (uncensored)
                    if at_risk[i] > 0:
                        survival_probs[i:] *= (at_risk[i] - 1) / at_risk[i]

            return survival_probs

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating survival probabilities: {e}")
            return np.ones(len(times))
    
    def _generate_candidate_labels(self, candidate_targets: List[Dict[str, Any]],
                                 horizons: Dict[str, Any], bars: pd.DataFrame,
                                 volatility_series: pd.Series, eligibility_mask: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate labels for all candidate targets."""
        try:
            candidate_labels = {}
            
            for candidate in candidate_targets:
                target_name = candidate['target_name']
                k_up = candidate['k_up']
                k_down = candidate['k_down']

                horizon_info = horizons.get(target_name, {}) if isinstance(horizons, dict) else {}
                if isinstance(horizon_info, dict):
                    horizon = horizon_info.get('horizon', self.config.min_horizon)
                else:
                    horizon = horizon_info or self.config.min_horizon

                # Generate labels with specific horizon
                labels = self._generate_labels_with_horizon(
                    k_up, k_down, horizon, bars, volatility_series, eligibility_mask
                )

                if not labels.empty:
                    if isinstance(horizon_info, dict):
                        labels.attrs['horizon_context'] = horizon_info
                    labels.attrs['horizon'] = horizon
                    candidate_labels[target_name] = labels

            return candidate_labels
            
        except Exception as e:
            tprint_error(f"❌ Error generating candidate labels: {e}")
            return {}
    
    def _generate_labels_with_horizon(self, k_up: float, k_down: float, horizon: int,
                                    bars: pd.DataFrame, volatility_series: pd.Series,
                                    eligibility_mask: pd.Series) -> pd.DataFrame:
        """Generate labels with specific horizon."""
        try:
            # Calculate target levels
            upper_targets = bars['close'] + k_up * volatility_series
            lower_targets = bars['close'] - k_down * volatility_series
            
            # Initialize labels
            labels = pd.Series(0, index=bars.index)
            confidence_scores = pd.Series(0.0, index=bars.index)
            sigma_payoffs = pd.Series(np.nan, index=bars.index, dtype=float)
            raw_payoffs = pd.Series(np.nan, index=bars.index, dtype=float)
            
            def _normalized_conf(distance: float, k_multiplier: float, sigma_scale: float) -> float:
                """Safely normalize confidence by volatility scale."""
                try:
                    if not np.isfinite(distance) or not np.isfinite(sigma_scale) or sigma_scale == 0:
                        return 0.0
                    denom = max(k_multiplier * sigma_scale, 1e-12)
                    return float(min(1.0, distance / denom))
                except Exception:
                    return 0.0

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

                raw_sigma = float(volatility_series.iloc[i]) if np.isscalar(volatility_series.iloc[i]) else float(volatility_series.iloc[i])
                if not np.isfinite(raw_sigma):
                    local_sigma = np.nan
                else:
                    local_sigma = raw_sigma
                sigma_scale = abs(local_sigma) if np.isfinite(local_sigma) and local_sigma != 0 else np.nan

                hit_direction = 0
                hit_index = None

                if upper_hits.any() and lower_hits.any():
                    # Both hit - check which comes first
                    upper_first_hit = upper_hits.idxmax() if upper_hits.any() else None
                    lower_first_hit = lower_hits.idxmax() if lower_hits.any() else None

                    if upper_first_hit is not None and lower_first_hit is not None:
                        if upper_first_hit <= lower_first_hit:
                            hit_direction = 1  # Upper hit first
                            hit_index = upper_first_hit
                            distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                            confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_down, sigma_scale)
                        else:
                            hit_direction = -1  # Lower hit first
                            hit_index = lower_first_hit
                            distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                            confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_up, sigma_scale)
                    elif upper_first_hit is not None:
                        hit_direction = 1
                        hit_index = upper_first_hit
                        distance_to_opposite = abs(future_prices.loc[upper_first_hit] - lower_target)
                        confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_down, sigma_scale)
                    elif lower_first_hit is not None:
                        hit_direction = -1
                        hit_index = lower_first_hit
                        distance_to_opposite = abs(future_prices.loc[lower_first_hit] - upper_target)
                        confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_up, sigma_scale)
                elif upper_hits.any():
                    hit_direction = 1
                    hit_index = upper_hits.idxmax()
                    distance_to_opposite = abs(future_prices.loc[hit_index] - lower_target)
                    confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_down, sigma_scale)
                elif lower_hits.any():
                    hit_direction = -1
                    hit_index = lower_hits.idxmax()
                    distance_to_opposite = abs(future_prices.loc[hit_index] - upper_target)
                    confidence_scores.iloc[i] = _normalized_conf(distance_to_opposite, k_up, sigma_scale)

                if hit_direction != 0 and hit_index is not None:
                    labels.iloc[i] = hit_direction
                    hit_price = future_prices.loc[hit_index]
                    payoff = hit_price - current_price
                    raw_payoffs.iloc[i] = payoff

                    if np.isfinite(sigma_scale) and sigma_scale != 0:
                        normalized_payoff = payoff / sigma_scale
                        if np.isfinite(normalized_payoff):
                            sigma_payoffs.iloc[i] = normalized_payoff
                        else:
                            sigma_payoffs.iloc[i] = np.nan
                    else:
                        sigma_payoffs.iloc[i] = np.nan
            
            # Create DataFrame with labels and confidence
            result_df = pd.DataFrame({
                'labels': labels,
                'confidence': confidence_scores,
                'sigma_payoff': sigma_payoffs,
                'raw_payoff': raw_payoffs
            }, index=bars.index)
            
            return result_df
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating labels with horizon: {e}")
            return pd.DataFrame()
    
    def _generate_confidence_features(self, bars: pd.DataFrame, volatility_series: pd.Series) -> pd.DataFrame:
        """Generate features for probabilistic confidence scoring."""
        try:
            features = pd.DataFrame(index=bars.index)
            
            # Price-based features
            features['returns'] = bars['close'].pct_change()
            features['volatility'] = volatility_series
            features['volatility_ratio'] = volatility_series / volatility_series.rolling(20).mean()
            
            # Volume features
            features['volume_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            features['volume_trend'] = bars['volume'].pct_change()
            
            # OHLC features
            features['high_low_ratio'] = (bars['high'] - bars['low']) / bars['close']
            features['close_open_ratio'] = (bars['close'] - bars['open']) / bars['open']
            
            # Technical indicators
            features['price_momentum'] = bars['close'] / bars['close'].shift(5) - 1
            features['volatility_momentum'] = volatility_series / volatility_series.shift(5) - 1
            
            # Fill NaN values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating confidence features: {e}")
            return pd.DataFrame()
    
    def _calculate_probabilistic_confidence(self, features: pd.Series, label: int, 
                                          volatility: float, hit_time: int) -> float:
        """Calculate probabilistic confidence using features."""
        try:
            # Simple logistic regression-like approach
            # In practice, this would be trained on historical data
            
            # Feature weights (would be learned from data)
            weights = {
                'returns': 0.3,
                'volatility': 0.2,
                'volatility_ratio': 0.15,
                'volume_ratio': 0.1,
                'high_low_ratio': 0.1,
                'close_open_ratio': 0.1,
                'price_momentum': 0.05
            }
            
            # Calculate weighted sum
            weighted_sum = 0.0
            for feature_name, weight in weights.items():
                if feature_name in features:
                    weighted_sum += weight * features[feature_name]
            
            # Apply sigmoid function
            confidence = 1 / (1 + np.exp(-weighted_sum))
            
            # Adjust for volatility (higher vol = lower confidence)
            vol_adjustment = 1 / (1 + volatility * 10)
            
            # Adjust for hit time (faster hits = higher confidence)
            time_adjustment = 1 / (1 + hit_time * 0.1)
            
            # Combine adjustments
            final_confidence = confidence * vol_adjustment * time_adjustment
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating probabilistic confidence: {e}")
            return 0.5
    
    def _resolve_label_conflicts(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve conflicts between different target labels at the same timestamp."""
        try:
            if labels_df.empty:
                return labels_df
            
            # Create conflict resolution rules
            # 1. Hierarchical precedence: small < medium < high
            # 2. Confidence-based selection within same level
            # 3. Multi-task output for complementary signals
            
            resolved_labels = labels_df.copy()
            
            # Group by target bands
            small_targets = [col for col in labels_df.columns if 'small' in col.lower()]
            medium_targets = [col for col in labels_df.columns if 'medium' in col.lower()]
            high_targets = [col for col in labels_df.columns if 'high' in col.lower()]
            
            # Apply hierarchical precedence
            for idx in labels_df.index:
                # Check for conflicts (multiple non-zero labels)
                non_zero_labels = labels_df.loc[idx][labels_df.loc[idx] != 0]
                
                if len(non_zero_labels) > 1:
                    # Apply hierarchical precedence: high > medium > small
                    target_to_keep = None

                    # Find highest precedence target that has a non-zero label
                    for col in non_zero_labels.index:
                        if col in high_targets:
                            target_to_keep = col
                            break
                        elif col in medium_targets:
                            target_to_keep = col
                            break
                        elif col in small_targets:
                            target_to_keep = col
                            break

                    # Zero out all other conflicting targets
                    if target_to_keep is not None:
                        for other_col in non_zero_labels.index:
                            if other_col != target_to_keep:
                                resolved_labels.loc[idx, other_col] = 0
            
            return resolved_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error resolving label conflicts: {e}")
            return labels_df
    
    def _apply_label_smoothing(self, labels_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
        """Apply label smoothing for better model calibration."""
        try:
            if labels_df.empty:
                return labels_df
            
            smoothed_labels = labels_df.copy()
            
            # Apply temporal smoothing to reduce micro-flips
            for col in labels_df.columns:
                if col in labels_df.columns:
                    labels_series = labels_df[col]
                    smoothed_series = self._temporal_smoothing(labels_series, confidence_df.get(col, pd.Series()))
                    smoothed_labels[col] = smoothed_series
            
            # Apply soft label smoothing (mix with uniform noise)
            if not confidence_df.empty:
                smoothed_labels = self._soft_label_smoothing(smoothed_labels, confidence_df)
            
            return smoothed_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error applying label smoothing: {e}")
            return labels_df
    
    def _temporal_smoothing(self, labels_series: pd.Series, confidence_series: pd.Series) -> pd.Series:
        """Apply temporal smoothing to reduce micro-flips."""
        try:
            if len(labels_series) < 3:
                return labels_series
            
            smoothed = labels_series.copy()
            window_size = 3  # Small window for temporal smoothing
            
            for i in range(window_size, len(labels_series)):
                # Get recent labels and confidences
                recent_labels = labels_series.iloc[i-window_size:i]
                recent_confidences = confidence_series.iloc[i-window_size:i] if not confidence_series.empty else pd.Series(1.0, index=recent_labels.index)
                
                # Weight by confidence
                if not recent_confidences.empty:
                    weights = recent_confidences / recent_confidences.sum()
                    weighted_labels = recent_labels * weights
                    smoothed_value = weighted_labels.sum()
                else:
                    smoothed_value = recent_labels.mean()
                
                # Apply smoothing only if confidence is high enough
                current_confidence = confidence_series.iloc[i] if not confidence_series.empty else 1.0
                if current_confidence > 0.5:
                    # Blend current label with smoothed value
                    alpha = 0.3  # Smoothing strength
                    smoothed.iloc[i] = (1 - alpha) * labels_series.iloc[i] + alpha * smoothed_value
            
            return smoothed
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in temporal smoothing: {e}")
            return labels_series
    
    def _soft_label_smoothing(self, labels_df: pd.DataFrame, confidence_df: pd.DataFrame) -> pd.DataFrame:
        """Apply soft label smoothing by mixing with uniform noise."""
        try:
            if labels_df.empty or confidence_df.empty:
                return labels_df
            
            smoothed_labels = labels_df.copy()
            smoothing_factor = 0.1  # 10% uniform noise
            
            for col in labels_df.columns:
                if col in labels_df.columns and col in confidence_df.columns:
                    labels_series = labels_df[col]
                    confidence_series = confidence_df[col]
                    
                    # Create soft labels
                    soft_labels = labels_series.copy()
                    
                    for i in range(len(labels_series)):
                        if confidence_series.iloc[i] > 0.5:  # Only smooth high-confidence labels
                            # Use deterministic smoothing instead of random noise for reproducibility
                            # Apply small amount of smoothing toward zero for regularization
                            soft_value = (1 - smoothing_factor) * labels_series.iloc[i] + smoothing_factor * 0.0

                            # Clamp to valid range
                            soft_labels.iloc[i] = max(-1, min(1, soft_value))
                    
                    smoothed_labels[col] = soft_labels
            
            return smoothed_labels
            
        except Exception as e:
            tprint_warning(f"⚠️ Error in soft label smoothing: {e}")
            return labels_df
    
    def _select_optimal_targets(self, candidate_labels: Dict[str, pd.DataFrame],
                              candidate_targets: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Select optimal targets based on quality and diversity with multiple-testing correction."""
        selection_metadata: Dict[str, Any] = {
            'total_candidates_evaluated': 0,
            'quality_scores': {},
            'quality_thresholds': {
                'base': self.config.min_lqs_score,
                'bonferroni': self.config.min_lqs_score,
                'benjamini_hochberg': {
                    'alpha': max(0.0, min(1.0, 1.0 - self.config.min_lqs_score)),
                    'critical_thresholds': {},
                    'accepted_count': 0,
                    'final_threshold': self.config.min_lqs_score,
                    'adjustment_applied': False,
                    'fallback_to_base_threshold': False,
                },
            },
            'correction_method': 'benjamini_hochberg',
            'qualified_targets_after_correction': 0,
        }

        try:
            if not candidate_labels:
                return {}, selection_metadata

            # Calculate quality scores for all candidates
            quality_scores: Dict[str, float] = {}
            total_candidates_evaluated = 0
            for target_name, labels_df in candidate_labels.items():
                if labels_df is None or labels_df.empty or 'labels' not in labels_df.columns:
                    continue

                labels = labels_df['labels']
                quality_score = self._calculate_target_quality_score(labels, pd.DataFrame(), pd.Series())
                quality_scores[target_name] = quality_score
                total_candidates_evaluated += 1

            selection_metadata['total_candidates_evaluated'] = total_candidates_evaluated
            selection_metadata['quality_scores'] = quality_scores

            if total_candidates_evaluated == 0:
                tprint_warning("⚠️ No candidate targets contained evaluable labels")
                return {}, selection_metadata

            base_threshold = self.config.min_lqs_score
            alpha = selection_metadata['quality_thresholds']['benjamini_hochberg']['alpha']
            bonferroni_threshold = self._calculate_bonferroni_threshold(base_threshold, total_candidates_evaluated)
            selection_metadata['quality_thresholds']['bonferroni'] = bonferroni_threshold

            # Sort by quality score (descending)
            sorted_scores = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)

            # Apply Benjamini-Hochberg correction on (1 - quality_score)
            bh_info = selection_metadata['quality_thresholds']['benjamini_hochberg']
            bh_info['critical_thresholds'] = {}
            accepted_count = 0

            if alpha > 0:
                for idx, (name, score) in enumerate(sorted_scores, start=1):
                    p_value = max(0.0, 1.0 - score)
                    critical_p = (idx / total_candidates_evaluated) * alpha
                    quality_cutoff = 1.0 - critical_p
                    bh_info['critical_thresholds'][name] = quality_cutoff
                    if p_value <= critical_p:
                        accepted_count = idx

            if accepted_count > 0:
                bh_info['accepted_count'] = accepted_count
                bh_info['adjustment_applied'] = True
                accepted_names = {
                    name for idx, (name, _score) in enumerate(sorted_scores, start=1)
                    if idx <= accepted_count
                }
                qualified_targets = {
                    name: score for name, score in sorted_scores
                    if name in accepted_names and score >= base_threshold
                }
                if qualified_targets:
                    min_selected_score = min(qualified_targets.values())
                else:
                    min_selected_score = base_threshold
                final_threshold = max(base_threshold, min_selected_score)
            else:
                qualified_targets = {
                    name: score for name, score in sorted_scores
                    if score >= base_threshold
                }
                final_threshold = base_threshold
                bh_info['fallback_to_base_threshold'] = True

            bh_info['final_threshold'] = final_threshold
            selection_metadata['quality_thresholds']['final'] = final_threshold
            selection_metadata['qualified_targets_after_correction'] = len(qualified_targets)

            if not qualified_targets:
                tprint_warning("⚠️ No targets passed quality threshold after correction")
                return {}, selection_metadata

            # Select targets by band
            selected_targets: Dict[str, Dict[str, Any]] = {}
            band_counts = {band: 0 for band in TargetBand}

            # Sort by quality score for selection ordering
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

            selection_metadata['selected_targets_after_constraints'] = len(selected_targets)

            # Ensure minimum targets
            if len(selected_targets) < self.config.min_targets_total:
                tprint_warning(
                    f"⚠️ Only {len(selected_targets)} targets selected, minimum is {self.config.min_targets_total}"
                )

            return selected_targets, selection_metadata

        except Exception as e:
            tprint_error(f"❌ Error selecting optimal targets: {e}")
            return {}, selection_metadata

    def _calculate_bonferroni_threshold(self, base_threshold: float, n_hypotheses: int) -> float:
        """Calculate Bonferroni-corrected threshold for quality scores."""
        if n_hypotheses <= 1:
            return base_threshold

        deficiency = max(0.0, 1.0 - base_threshold)
        adjusted_threshold = 1.0 - (deficiency / float(n_hypotheses))
        return float(np.clip(adjusted_threshold, 0.0, 1.0))
    
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
            sigma_payoff_df = pd.DataFrame(index=bars.index)
            raw_payoff_df = pd.DataFrame(index=bars.index)
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
                    if 'sigma_payoff' in target_result:
                        sigma_payoff_df[target_name] = target_result['sigma_payoff']
                    if 'raw_payoff' in target_result:
                        raw_payoff_df[target_name] = target_result['raw_payoff']
                    eligibility_df[f"{target_name}_eligibility"] = eligibility_mask

            return {
                'labels': labels_df,
                'confidence_scores': confidence_df,
                'sigma_payoffs': sigma_payoff_df,
                'raw_payoffs': raw_payoff_df,
                'eligibility_masks': eligibility_df
            }

        except Exception as e:
            tprint_error(f"❌ Error generating final labels: {e}")
            return {
                'labels': pd.DataFrame(),
                'confidence_scores': pd.DataFrame(),
                'sigma_payoffs': pd.DataFrame(),
                'raw_payoffs': pd.DataFrame(),
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