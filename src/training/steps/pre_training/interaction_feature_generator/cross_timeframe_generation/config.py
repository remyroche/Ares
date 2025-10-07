"""Configuration dataclasses for the cross-timeframe pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Tuple


@dataclass
class SessionConfig:
    """Configuration related to session handling and resampling."""

    base_timeframe_minutes: int = 5
    session_start_hour: int = 9
    session_end_hour: int = 16
    dst_handling: bool = True
    market_timezone: str | None = None


@dataclass
class ProbeConfig:
    """Configuration for the Phase-1 probing stage."""

    coarse_grid_min: int = 15
    coarse_grid_max: int = 298
    adaptive_refinement_threshold: float = 0.75


@dataclass
class OptimizationConfig:
    """Configuration for Phase-2 optimization."""

    local_grid_factor: float = 0.5
    ic_surface_smoothing: str = "spline"


@dataclass
class RegimeConfig:
    """Configuration for regime segmentation and monitoring triggers."""

    change_point_method: str = "PELT"
    regime_vol_quantile: float = 0.6
    bocpd_hazard: float = 1 / 200


@dataclass
class ScoringConfig:
    """Configuration for adaptive scoring and penalty learning."""

    lambda_unc: float = 0.10
    lambda_cost: float = 0.05
    lambda_stale: float = 0.05
    meta_learning_range: float = 0.05


@dataclass
class AssignmentConfig:
    """Configuration for EHU/RIH assignment."""

    rih_threshold: float = 0.01
    hybrid_mode: bool = True


@dataclass
class SelectionConfig:
    """Configuration for both knapsack and statistical selection stages."""

    max_cost_ms: float = 25.0
    max_features: int = 120
    max_correlation: float = 0.8
    min_samples_for_correlation: int = 30

    stability_resamples: int = 80
    fdr_q: float = 0.1
    min_conditional_ic: float = 0.25
    enable_group_lasso: bool = False
    enable_cross_asset: bool = False
    cross_asset_lags: Tuple[int, ...] | None = None


@dataclass
class EvaluationConfig:
    """Configuration for the walk-forward evaluation stage."""

    embargo_minutes: int = 60
    walk_forward_folds: int = 5
    spa_test: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and automation."""

    adaptive_penalties: bool = True
    dashboard_enabled: bool = True


@dataclass
class PipelineConfig:
    """Composite configuration for the cross-timeframe pipeline."""

    session: SessionConfig = field(default_factory=SessionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def __getattr__(self, item: str) -> Any:
        """Provide backward-compatible attribute access to segment fields."""

        for segment in self._iter_segments():
            if hasattr(segment, item):
                return getattr(segment, item)
        raise AttributeError(f"{item} is not a valid configuration attribute")

    def __setattr__(self, key: str, value: Any) -> None:
        if key in {"session", "probe", "optimization", "regime", "scoring",
                   "assignment", "selection", "evaluation", "monitoring"}:
            super().__setattr__(key, value)
            return

        for segment in self._iter_segments():
            if hasattr(segment, key):
                setattr(segment, key, value)
                return

        super().__setattr__(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration into a nested dictionary."""

        return {
            name: self._segment_to_dict(segment)
            for name, segment in self._iter_named_segments()
        }

    def _iter_segments(self) -> Iterable[Any]:
        return (
            self.session,
            self.probe,
            self.optimization,
            self.regime,
            self.scoring,
            self.assignment,
            self.selection,
            self.evaluation,
            self.monitoring,
        )

    def _iter_named_segments(self) -> Iterator[Tuple[str, Any]]:
        yield from (
            ("session", self.session),
            ("probe", self.probe),
            ("optimization", self.optimization),
            ("regime", self.regime),
            ("scoring", self.scoring),
            ("assignment", self.assignment),
            ("selection", self.selection),
            ("evaluation", self.evaluation),
            ("monitoring", self.monitoring),
        )

    @staticmethod
    def _segment_to_dict(segment: Any) -> Dict[str, Any]:
        if hasattr(segment, "__dict__"):
            return {k: getattr(segment, k) for k in vars(segment)}
        return dict(segment)
