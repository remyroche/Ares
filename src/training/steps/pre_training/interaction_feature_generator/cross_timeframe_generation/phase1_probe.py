"""
Phase-1 HTF Probe Stage

Implements coarse, adaptive grid generation for HTF features with:
- Coarse grids per family (Trend/Level & Vol, Osc, Anchor)
- Adaptive refinement based on top-quartile performance
- Regime-aware scoring with change-point handling
- Early stopping and shortlisting
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from itertools import product

# Import existing components
import sys
sys.path.append('src/training/steps/pre_training/interaction_feature_generator/feature_interaction_generation')
from feature_engineering_roadmap.feature_registry import FeatureRegistry
from feature_engineering_roadmap.transforms import TransformRouter, create_default_transform_config
from .htf_utils import (
    build_htf_family_catalog,
    format_transform_suffix,
    resample_htf_series,
)
from ..feature_interaction_generation.feature_engineering import (
    FeatureRegistry,
    FeatureFamily,
    TransformRouter,
    create_default_transform_config,
)

from .scoring_system import AdaptiveScoringSystem
from . import htf_base_features
from .config import ProbeConfig, SessionConfig


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

    def __init__(self, config: ProbeConfig):
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

    def __init__(self, session_config: SessionConfig):
        self.session_config = session_config
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

            base_feature=base_feature,
            lookback_minutes=lookback_minutes,
            htf_series=htf_series,
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


class FamilyProcessingError(RuntimeError):
    """Raised when an HTF family fails to yield any viable candidates."""

    def __init__(self, family: str, details: Optional[List[str]] = None):
        self.family = family
        self.details = details or []

        message = f"Failed to produce any valid candidates for family '{family}'."
        if self.details:
            message = f"{message} {' '.join(self.details)}"

        super().__init__(message)


class Phase1HTFProbe:
    """Phase-1 HTF probe stage implementation."""

    def __init__(
        self,
        probe_config: ProbeConfig,
        session_config: SessionConfig,
        scoring_system: Optional[AdaptiveScoringSystem] = None,
    ):
        self.probe_config = probe_config
        self.session_config = session_config
        self.logger = logging.getLogger(__name__)

        self.grid_generator = CoarseGridGenerator(probe_config)
        self.htf_generator = HTFFeatureGenerator(session_config)
        self.scoring_system: Optional[AdaptiveScoringSystem] = None

        if scoring_system is not None:
            self.set_scoring_system(scoring_system)

    def set_scoring_system(self, scoring_system: AdaptiveScoringSystem) -> None:
        """Inject the centralized adaptive scoring system."""
        self.scoring_system = scoring_system

    def _ensure_scoring_system(self) -> AdaptiveScoringSystem:
        """Return the adaptive scoring system or raise a helpful error."""
        if self.scoring_system is None:
            raise RuntimeError(
                "Phase1HTFProbe requires an adaptive scoring system before scoring. "
                "Provide one via the constructor or set_scoring_system()."
            )

        return self.scoring_system
        
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

        self._ensure_scoring_system()
        
        results = {
            'candidates': [],
            'family_performance': {},
            'shortlisted_candidates': [],
            'early_stopped_families': []
        }
        
        # Process each HTF family
        for family, base_features in self.htf_generator.htf_families.items():
            self.logger.info(f"Processing family: {family}")

            try:
                family_results = self._process_family(
                    family, base_features, sessionized_data, regime_segments, targets
                )
            except FamilyProcessingError as exc:
                self.logger.error(
                    "Phase-1 probe aborted while processing family '%s': %s",
                    family,
                    exc,
                )
                raise

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
        errors: List[str] = []
        empty_attempts: List[str] = []
        has_success = False

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
                    has_success = True
                    self.logger.debug(
                        "Scored %s@%s for %d regime variants",
                        base_feature,
                        lookback,
                        len(regime_candidates)
                    )
                else:
                    empty_attempts.append(f"{base_feature}@{lookback}")

            except Exception as e:
                self.logger.warning(f"Failed to process {base_feature}@{lookback}: {e}")
                errors.append(f"{base_feature}@{lookback}: {e}")
                continue

        if not has_success:
            detail_parts: List[str] = []
            if errors:
                detail_parts.append(
                    "Encountered errors for "
                    + ", ".join(errors)
                )
            if empty_attempts:
                detail_parts.append(
                    "No viable regimes produced for "
                    + ", ".join(empty_attempts)
                )
            if not detail_parts:
                detail_parts.append("No candidates evaluated for generated grid.")

            detail_message = " ".join(detail_parts)
            self.logger.error(
                "Family '%s' produced no successful candidates. %s",
                family,
                detail_message,
            )
            raise FamilyProcessingError(family, detail_parts)

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
        min_segment_points = max(
            50,
            int(self.session_config.base_timeframe_minutes * 4),
        )

        scoring_system = self._ensure_scoring_system()

        def _build_candidate(segment_data: pd.DataFrame,
                             regime_label: str,
                             segment_index: Optional[int],
                             segment_meta: Dict[str, Any]) -> Optional[HTFCandidate]:
            if len(segment_data) < min_segment_points:
                return None

            feature_slice = segment_data['htf_feature']
            target_slice = segment_data['target']

            scoring_result = scoring_system.score_feature_candidate(
                feature=feature_slice,
                target=target_slice,
                lookback=lookback,
                family=family,
                regime=regime_label,
                regime_segments=segments,
            )

            if scoring_result is None:
                return None

            segment_info = {
                'segment_index': segment_index,
                **segment_meta
            }

            performance_metadata = {
                'utility_score': scoring_result.utility_score,
                'ic_oos': scoring_result.ic_oos,
                'se_wild_bootstrap': scoring_result.se_wild_bootstrap,
                'se_stationary_bootstrap': scoring_result.se_stationary_bootstrap,
                'fold_pass_rate': scoring_result.fold_pass_rate,
                'cpu_p95': scoring_result.cpu_p95,
                'staleness': scoring_result.staleness,
                'regime_weight': scoring_result.regime_weight,
            }

            metadata: Dict[str, Any] = {
                'regime_segment': segment_info,
                'performance': performance_metadata,
            }

            if scoring_result.metadata:
                metadata['scoring_metadata'] = scoring_result.metadata

            staleness_summary = self._get_staleness_summary(base_feature, lookback, family)
            if staleness_summary is not None:
                metadata['staleness_summary'] = staleness_summary

            return HTFCandidate(
                family=family,
                base_feature=base_feature,
                lookback_minutes=lookback,
                regime=regime_label,
                utility_score=scoring_result.utility_score,
                ic_oos=scoring_result.ic_oos,
                se_wild_bootstrap=scoring_result.se_wild_bootstrap,
                cpu_p95=scoring_result.cpu_p95,
                staleness=scoring_result.staleness,
                fold_pass_rate=scoring_result.fold_pass_rate,
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

    def _get_staleness_summary(
        self,
        base_feature: str,
        lookback: int,
        family: str,
    ) -> Optional[Any]:
        """Fetch staleness curve summary for metadata enrichment."""

        scoring_system = self._ensure_scoring_system()
        staleness_calculator = getattr(scoring_system, 'staleness_calculator', None)
        if staleness_calculator is None:
            return None

        curve_calculator = getattr(staleness_calculator, 'curve_calculator', None)
        if curve_calculator is None:
            return None

        base_timeframe = getattr(self.session_config, 'base_timeframe_minutes', 5)
        staleness_value = staleness_calculator.calculate_staleness(
            lookback=lookback,
            family=family,
            base_timeframe=base_timeframe,
        )

        try:
            return curve_calculator.get_summary(
                feature_name=base_feature,
                family=family,
                lookback=lookback,
                base_timeframe=base_timeframe,
            )
        except Exception as exc:
            self.logger.debug(
                "Failed to fetch staleness summary for %s@%s (%s): %s",
                base_feature,
                lookback,
                family,
                exc,
            )
            return None
    
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