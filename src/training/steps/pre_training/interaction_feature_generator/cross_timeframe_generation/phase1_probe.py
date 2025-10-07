"""
Phase-1 HTF Probe Stage

Implements coarse, adaptive grid generation for HTF features with:
- Coarse grids per family (Trend/Level & Vol, Osc, Anchor)
- Adaptive refinement based on top-quartile performance
- Regime-aware scoring with change-point handling
- Early stopping and shortlisting
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import product
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from .staleness_curve import StalenessCurveCalculator

# Import existing components
import sys
sys.path.append('src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation')
from feature_engineering.feature_registry import FeatureRegistry
from feature_engineering.transforms import TransformRouter, create_default_transform_config
from .htf_utils import (
    build_htf_family_catalog,
    format_transform_suffix,
    resample_htf_series,
)

from .scoring_system import AdaptiveScoringSystem


@dataclass
class HTFCandidate:
    """Represents an HTF feature candidate."""
    family: str
    base_feature: str
    lookback_minutes: int
    regime: str
    utility_score: float
    ic_oos: float
    se_wild_bootstrap: float
    cpu_p95: float
    staleness: float
    fold_pass_rate: float
    metadata: Dict[str, Any]


class CoarseGridGenerator:
    """Generates coarse adaptive grids for HTF features."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Base coarse grid (15m to 300m)
        self.base_coarse_grid = self._generate_coarse_grid()
        
    def _generate_coarse_grid(self) -> List[int]:
        """Generate base coarse grid in minutes."""
        # Log-spaced grid from 15m to 300m
        min_minutes = self.config.coarse_grid_min
        max_minutes = self.config.coarse_grid_max
        
        # Create log-spaced grid
        n_points = 8  # Start with 8 points
        log_min = np.log(min_minutes)
        log_max = np.log(max_minutes)
        log_points = np.linspace(log_min, log_max, n_points)
        
        grid = np.exp(log_points).astype(int)
        return sorted(list(set(grid)))  # Remove duplicates and sort
    
    def generate_adaptive_grid(self, 
                             family: str, 
                             performance_history: Dict[int, float]) -> List[int]:
        """
        Generate adaptive grid based on performance history.
        
        Args:
            family: HTF family name
            performance_history: Dict mapping lookback -> utility score
            
        Returns:
            List of lookback values to probe
        """
        base_grid = self.base_coarse_grid.copy()
        
        # Find top-quartile performers
        if performance_history:
            scores = list(performance_history.values())
            threshold = np.percentile(scores, 75)
            
            # Add neighbors for top-quartile performers
            for lookback, score in performance_history.items():
                if score >= threshold:
                    # Add neighbors (e.g., 45/90m around 60)
                    neighbors = self._generate_neighbors(lookback)
                    base_grid.extend(neighbors)
        
        # Remove duplicates and sort
        return sorted(list(set(base_grid)))
    
    def _generate_neighbors(self, lookback: int) -> List[int]:
        """Generate neighbor lookbacks around a given value."""
        # Generate neighbors with 0.75x and 1.33x factors
        neighbors = [
            int(lookback * 0.75),
            int(lookback * 1.33)
        ]
        
        # Add some additional nearby values
        neighbors.extend([
            lookback - 15,
            lookback + 15,
            lookback - 30,
            lookback + 30
        ])
        
        # Filter to valid range
        min_val = self.config.coarse_grid_min
        max_val = self.config.coarse_grid_max
        return [n for n in neighbors if min_val <= n <= max_val]


class HTFFeatureGenerator:
    """Generates HTF features from base features."""

    def __init__(self, config):
        self.config = config
        self.feature_registry = FeatureRegistry()
        self.logger = logging.getLogger(__name__)
        self.htf_families, self.base_feature_to_family = build_htf_family_catalog(
            self.feature_registry
        )

    def generate_htf_feature(self,
                           data: pd.DataFrame,
                           base_feature: str,
                           lookback_minutes: int,
                           family: str) -> pd.Series:
        """
        Generate HTF feature by resampling base feature to higher timeframe.
        
        Args:
            data: OHLCV data
            base_feature: Base feature name
            lookback_minutes: HTF lookback in minutes
            family: Feature family
            
        Returns:
            HTF feature series
        """
        if base_feature not in self.base_feature_to_family:
            raise ValueError(f"Base feature '{base_feature}' is not registered for HTF usage")

        expected_family = self.base_feature_to_family[base_feature]
        if family != expected_family:
            self.logger.warning(
                "Family mismatch for %s: expected %s, received %s",
                base_feature,
                expected_family,
                family,
            )

        metadata = self.feature_registry.get_feature_metadata(base_feature)
        base_series = self.feature_registry.compute_feature(base_feature, data)

        htf_series = resample_htf_series(base_series, lookback_minutes, metadata.family)

        transformed_series = self._apply_transforms(
            base_feature,
            lookback_minutes,
            htf_series,
        )

        return transformed_series

    def _apply_transforms(
        self,
        base_feature: str,
        lookback_minutes: int,
        htf_series: pd.Series,
    ) -> pd.Series:
        """Apply the default transform pipeline to a resampled HTF series."""
        transform_config = create_default_transform_config([base_feature])
        transform_router = TransformRouter(transform_config)

        transformed = transform_router.fit_transform(
            pd.DataFrame({base_feature: htf_series}),
            pd.DataFrame({base_feature: htf_series}),
        )

        transformed_df = transformed.get(base_feature, {}).get('train')
        if transformed_df is None or transformed_df.empty:
            return pd.Series(index=htf_series.index, dtype=float)

        transformed_series = transformed_df.iloc[:, 0]
        suffix = format_transform_suffix(transform_config[base_feature])
        transformed_series.name = f"t/{base_feature}_htf{lookback_minutes}/{suffix}"
        return transformed_series


class Phase1HTFProbe:
    """Phase-1 HTF probe stage implementation."""

    def __init__(self, config, scoring_system: Optional[AdaptiveScoringSystem] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.grid_generator = CoarseGridGenerator(config)
        self.htf_generator = HTFFeatureGenerator(config)
        self.scoring_system = scoring_system

    def set_scoring_system(self, scoring_system: AdaptiveScoringSystem) -> None:
        """Inject the centralized adaptive scoring system."""
        self.scoring_system = None  # Will be injected
        self.staleness_calculator = StalenessCurveCalculator(
            default_base_timeframe=getattr(config, 'base_timeframe_minutes', 5)
        )
        
    def run_probe_stage(self, 
                       sessionized_data: Dict[str, Any],
                       regime_segments: Dict[str, Any],
                       targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run Phase-1 HTF probe stage.
        
        Args:
            sessionized_data: Sessionized and aligned data
            regime_segments: Regime segmentation results
            targets: Target variables
            
        Returns:
            Phase-1 results with shortlisted HTF candidates
        """
        self.logger.info("Starting Phase-1 HTF probe stage")
        
        results = {
            'candidates': [],
            'family_performance': {},
            'shortlisted_candidates': [],
            'early_stopped_families': []
        }
        
        # Process each HTF family
        for family, base_features in self.htf_generator.htf_families.items():
            self.logger.info(f"Processing family: {family}")
            
            family_results = self._process_family(
                family, base_features, sessionized_data, regime_segments, targets
            )
            
            results['candidates'].extend(family_results['candidates'])
            results['family_performance'][family] = family_results['performance']
            
            if family_results['early_stopped']:
                results['early_stopped_families'].append(family)
            else:
                results['shortlisted_candidates'].extend(family_results['shortlisted'])
        
        # Apply early stopping across families
        results = self._apply_early_stopping(results)
        
        self.logger.info(f"Phase-1 completed: {len(results['shortlisted_candidates'])} candidates shortlisted")
        return results
    
    def _process_family(self,
                       family: str,
                       base_features: List[str],
                       sessionized_data: Dict[str, Any],
                       regime_segments: Dict[str, Any],
                       targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Process a single HTF family."""
        
        # Generate adaptive grid for family
        performance_history = {}  # Would be populated from previous runs
        grid = self.grid_generator.generate_adaptive_grid(family, performance_history)
        
        candidates = []
        family_scores = []
        
        # Test each combination of base feature and lookback
        for base_feature, lookback in product(base_features, grid):
            try:
                # Generate HTF feature
                htf_feature = self.htf_generator.generate_htf_feature(
                    sessionized_data['aligned_data'],
                    base_feature,
                    lookback,
                    family
                )

                # Score the candidate
                regime_candidates = self._score_candidate(
                    htf_feature, base_feature, lookback, family,
                    regime_segments, targets
                )

                if regime_candidates:
                    candidates.extend(regime_candidates)
                    family_scores.extend([c.utility_score for c in regime_candidates])
                    self.logger.debug(
                        "Scored %s@%s for %d regime variants",
                        base_feature,
                        lookback,
                        len(regime_candidates)
                    )

            except Exception as e:
                self.logger.warning(f"Failed to process {base_feature}@{lookback}: {e}")
                continue
        
        # Check for early stopping
        early_stopped = self._check_early_stopping(family_scores)
        
        # Shortlist top candidates
        if not early_stopped:
            shortlisted = self._shortlist_candidates(candidates, family)
        else:
            shortlisted = []
        
        return {
            'candidates': candidates,
            'performance': family_scores,
            'shortlisted': shortlisted,
            'early_stopped': early_stopped
        }
    
    def _score_candidate(self,
                        htf_feature: pd.Series,
                        base_feature: str,
                        lookback: int,
                        family: str,
                        regime_segments: Dict[str, Any],
                        targets: Optional[pd.Series] = None) -> List[HTFCandidate]:
        """Score an HTF candidate across available regimes."""

        if targets is None or len(htf_feature) == 0:
            return []

        # Align features and targets
        aligned_data = pd.DataFrame({
            'htf_feature': htf_feature,
            'target': targets
        }).dropna()

        if len(aligned_data) < 100:  # Need sufficient data across the full window
            return []

        segments = (regime_segments or {}).get('segments', [])
        min_segment_points = max(50, int(self.config.base_timeframe_minutes * 4))

        def _build_candidate(segment_data: pd.DataFrame,
                             regime_label: str,
                             segment_index: Optional[int],
                             segment_meta: Dict[str, Any]) -> Optional[HTFCandidate]:
            if len(segment_data) < min_segment_points:
                return None

            feature_slice = segment_data['htf_feature']
            target_slice = segment_data['target']

            ic_oos = self._calculate_ic(feature_slice, target_slice)
            se_wild_bootstrap = self._calculate_wild_bootstrap_se(feature_slice, target_slice)
            cpu_p95 = self._estimate_cpu_cost(lookback, family)
            staleness = self._calculate_staleness(lookback, family)
            fold_pass_rate = self._calculate_fold_pass_rate(feature_slice, target_slice)

            utility_score = self._calculate_utility_score(
                ic_oos, se_wild_bootstrap, cpu_p95, staleness
            )

            metadata = {
                'regime_segment': {
                    'segment_index': segment_index,
                    **segment_meta
                },
                'performance': {
                    'ic_oos': ic_oos,
                    'utility_score': utility_score,
                    'se_wild_bootstrap': se_wild_bootstrap,
                    'fold_pass_rate': fold_pass_rate
                }
            }

            return HTFCandidate(
                family=family,
                base_feature=base_feature,
                lookback_minutes=lookback,
                regime=regime_label,
                utility_score=utility_score,
                ic_oos=ic_oos,
                se_wild_bootstrap=se_wild_bootstrap,
                cpu_p95=cpu_p95,
                staleness=staleness,
                fold_pass_rate=fold_pass_rate,
                metadata=metadata
            )

        candidates: List[HTFCandidate] = []

        if segments:
            for idx, segment in enumerate(segments):
                start_time = getattr(segment, 'start_time', None)
                end_time = getattr(segment, 'end_time', None)
                if start_time is None or end_time is None:
                    continue

                mask = (aligned_data.index >= start_time) & (aligned_data.index <= end_time)
                segment_data = aligned_data.loc[mask]

                segment_meta = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'segment_length': len(segment_data),
                    'regime_type': getattr(segment, 'regime_type', None),
                    'volatility_level': getattr(segment, 'volatility_level', None),
                    'mean_return': getattr(segment, 'mean_return', None)
                }

                regime_label = getattr(segment, 'regime_type', f'regime_{idx}')
                candidate = _build_candidate(segment_data, regime_label, idx, segment_meta)
                if candidate:
                    candidates.append(candidate)

        if not candidates:
            # Fall back to mixed regime scoring if no segments or all were insufficient
            segment_meta = {
                'start_time': aligned_data.index.min(),
                'end_time': aligned_data.index.max(),
                'segment_length': len(aligned_data),
                'regime_type': 'mixed',
                'volatility_level': None,
                'mean_return': None
            }
            fallback_candidate = _build_candidate(
                aligned_data,
                'mixed',
                None,
                segment_meta
            )
            if fallback_candidate:
                candidates.append(fallback_candidate)

        return candidates
    
    def _calculate_ic(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Coefficient."""
        correlation = feature.corr(target)
        return correlation if not pd.isna(correlation) else 0.0
    
    def _calculate_wild_bootstrap_se(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate wild bootstrap standard error."""
        # Simplified implementation
        n = len(feature)
        if n < 10:
            return 1.0
        
        # Wild bootstrap with Rademacher weights
        n_bootstrap = 100
        correlations = []
        
        for _ in range(n_bootstrap):
            weights = np.random.choice([-1, 1], size=n)
            weighted_feature = feature * weights
            corr = weighted_feature.corr(target)
            if not pd.isna(corr):
                correlations.append(corr)
        
        return np.std(correlations) if correlations else 1.0
    
    def _estimate_cpu_cost(self, lookback: int, family: str) -> float:
        """Estimate CPU cost in milliseconds."""
        # Base cost per lookback minute
        base_cost = 0.01  # ms per minute
        
        # Family-specific multipliers
        family_multipliers = {
            'trend_level_vol': 1.0,
            'oscillators': 1.2,
            'anchors': 0.8
        }
        
        multiplier = family_multipliers.get(family, 1.0)
        return base_cost * lookback * multiplier
    
    def _calculate_fold_pass_rate(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate fold pass rate using time series cross-validation."""
        if len(feature) < 50:
            return 0.0
        
        # Use 3-fold time series split
        tscv = TimeSeriesSplit(n_splits=3)
        pass_count = 0
        total_folds = 0
        
        for train_idx, val_idx in tscv.split(feature):
            if len(val_idx) < 10:
                continue
                
            train_feature = feature.iloc[train_idx]
            train_target = target.iloc[train_idx]
            val_feature = feature.iloc[val_idx]
            val_target = target.iloc[val_idx]
            
            # Calculate IC on validation set
            val_ic = val_feature.corr(val_target)
            
            # Pass if IC > 0.05
            if not pd.isna(val_ic) and val_ic > 0.05:
                pass_count += 1
            
            total_folds += 1
        
        return pass_count / total_folds if total_folds > 0 else 0.0
    
    def _calculate_utility_score(self,
                               ic_oos: float,
                               se_wild_bootstrap: float,
                               cpu_p95: float,
                               staleness: float) -> float:
        """Calculate utility score using the centralized scoring system."""
        if self.scoring_system is None:
            raise ValueError("Adaptive scoring system must be provided to Phase1HTFProbe")

        return self.scoring_system.calculate_utility_score(
            ic_oos=ic_oos,
            se_wild_bootstrap=se_wild_bootstrap,
            cpu_p95=cpu_p95,
            staleness=staleness,
        )
    
    def _check_early_stopping(self, family_scores: List[float]) -> bool:
        """Check if family should be early stopped."""
        if not family_scores:
            return True
        
        # Early stop if all scores < 0
        return all(score < 0 for score in family_scores)
    
    def _shortlist_candidates(self, 
                             candidates: List[HTFCandidate],
                             family: str) -> List[HTFCandidate]:
        """Shortlist top candidates for a family."""
        if not candidates:
            return []
        
        # Sort by utility score
        sorted_candidates = sorted(candidates, key=lambda x: x.utility_score, reverse=True)
        
        # Keep top 2 with positive utility and fold pass rate >= 60%
        shortlisted = []
        for candidate in sorted_candidates:
            if (candidate.utility_score > 0 and 
                candidate.fold_pass_rate >= 0.6 and 
                len(shortlisted) < 2):
                shortlisted.append(candidate)
        
        return shortlisted
    
    def _apply_early_stopping(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply early stopping across families."""
        # If all families are early stopped, keep the best performing one
        if len(results['early_stopped_families']) == len(self.htf_generator.htf_families):
            # Find the family with the best performance
            best_family = None
            best_score = -np.inf
            
            for family, performance in results['family_performance'].items():
                if performance:
                    max_score = max(performance)
                    if max_score > best_score:
                        best_score = max_score
                        best_family = family
            
            if best_family:
                # Remove from early stopped and add to shortlisted
                results['early_stopped_families'] = [
                    f for f in results['early_stopped_families'] if f != best_family
                ]
                # Add best candidates from this family
                family_candidates = [
                    c for c in results['candidates'] 
                    if c.family == best_family
                ]
                results['shortlisted_candidates'].extend(
                    self._shortlist_candidates(family_candidates, best_family)
                )
        
        return results