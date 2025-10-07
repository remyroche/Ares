"""Configuration objects for the cross-timeframe pipeline.

The previous implementation relied on a single ``PipelineConfig`` dataclass
that was passed wholesale into every component.  This made it difficult to
reason about component boundaries, complicated testing, and encouraged
unintended coupling across modules.  The new configuration module introduces
focused configuration dataclasses for each subsystem while keeping
``PipelineConfig`` as a convenient aggregate with backwards-compatible access
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionConfig:
    """Sessionization and base timeframe configuration."""

    base_timeframe_minutes: int = 5
    session_start_hour: int = 9
    session_end_hour: int = 16
    dst_handling: bool = True


@dataclass
class ProbeConfig:
    """Configuration for coarse HTF probing."""

    coarse_grid_min: int = 15
    coarse_grid_max: int = 298
    adaptive_refinement_threshold: float = 0.75


@dataclass
class OptimizationConfig:
    """Configuration for the Phase-2 optimization stage."""

    local_grid_factor: float = 0.5
    ic_surface_smoothing: str = "spline"


@dataclass
class RegimeConfig:
    """Regime segmentation configuration."""

    change_point_method: str = "PELT"
    regime_vol_quantile: float = 0.6
    bocpd_hazard: float = 1 / 200


@dataclass
class ScoringConfig:
    """Adaptive scoring configuration."""

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
    """Configuration shared by knapsack and statistical selection."""

    max_cost_ms: float = 25.0
    max_features: int = 120
    max_correlation: float = 0.8
    stability_resamples: int = 80
    fdr_q: float = 0.1
    min_conditional_ic: float = 0.25
    min_samples_for_correlation: int = 30


@dataclass
class EvaluationConfig:
    """Walk-forward evaluation configuration."""

    embargo_minutes: int = 60
    walk_forward_folds: int = 5
    spa_test: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and automation configuration."""

    adaptive_penalties: bool = True
    dashboard_enabled: bool = True


@dataclass(init=False)
class PipelineConfig:
    """Aggregate configuration composed of focused config segments."""

    session: SessionConfig = field(default_factory=SessionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def __init__(
        self,
        *,
        session: Optional[SessionConfig] = None,
        probe: Optional[ProbeConfig] = None,
        optimization: Optional[OptimizationConfig] = None,
        regime: Optional[RegimeConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        assignment: Optional[AssignmentConfig] = None,
        selection: Optional[SelectionConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        monitoring: Optional[MonitoringConfig] = None,
        **overrides: Any,
    ) -> None:
        self.session = session or SessionConfig()
        self.probe = probe or ProbeConfig()
        self.optimization = optimization or OptimizationConfig()
        self.regime = regime or RegimeConfig()
        self.scoring = scoring or ScoringConfig()
        self.assignment = assignment or AssignmentConfig()
        self.selection = selection or SelectionConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.monitoring = monitoring or MonitoringConfig()

        for key, value in overrides.items():
            if not hasattr(self, key):
                raise TypeError(f"Unknown configuration option '{key}'")
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Convenience helpers mirroring the legacy flat PipelineConfig API.
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default

    @property
    def base_timeframe_minutes(self) -> int:
        return self.session.base_timeframe_minutes

    @base_timeframe_minutes.setter
    def base_timeframe_minutes(self, value: int) -> None:
        self.session.base_timeframe_minutes = value

    @property
    def session_start_hour(self) -> int:
        return self.session.session_start_hour

    @session_start_hour.setter
    def session_start_hour(self, value: int) -> None:
        self.session.session_start_hour = value

    @property
    def session_end_hour(self) -> int:
        return self.session.session_end_hour

    @session_end_hour.setter
    def session_end_hour(self, value: int) -> None:
        self.session.session_end_hour = value

    @property
    def dst_handling(self) -> bool:
        return self.session.dst_handling

    @dst_handling.setter
    def dst_handling(self, value: bool) -> None:
        self.session.dst_handling = value

    @property
    def coarse_grid_min(self) -> int:
        return self.probe.coarse_grid_min

    @coarse_grid_min.setter
    def coarse_grid_min(self, value: int) -> None:
        self.probe.coarse_grid_min = value

    @property
    def coarse_grid_max(self) -> int:
        return self.probe.coarse_grid_max

    @coarse_grid_max.setter
    def coarse_grid_max(self, value: int) -> None:
        self.probe.coarse_grid_max = value

    @property
    def adaptive_refinement_threshold(self) -> float:
        return self.probe.adaptive_refinement_threshold

    @adaptive_refinement_threshold.setter
    def adaptive_refinement_threshold(self, value: float) -> None:
        self.probe.adaptive_refinement_threshold = value

    @property
    def local_grid_factor(self) -> float:
        return self.optimization.local_grid_factor

    @local_grid_factor.setter
    def local_grid_factor(self, value: float) -> None:
        self.optimization.local_grid_factor = value

    @property
    def ic_surface_smoothing(self) -> str:
        return self.optimization.ic_surface_smoothing

    @ic_surface_smoothing.setter
    def ic_surface_smoothing(self, value: str) -> None:
        self.optimization.ic_surface_smoothing = value

    @property
    def change_point_method(self) -> str:
        return self.regime.change_point_method

    @change_point_method.setter
    def change_point_method(self, value: str) -> None:
        self.regime.change_point_method = value

    @property
    def regime_vol_quantile(self) -> float:
        return self.regime.regime_vol_quantile

    @regime_vol_quantile.setter
    def regime_vol_quantile(self, value: float) -> None:
        self.regime.regime_vol_quantile = value

    @property
    def bocpd_hazard(self) -> float:
        return self.regime.bocpd_hazard

    @bocpd_hazard.setter
    def bocpd_hazard(self, value: float) -> None:
        self.regime.bocpd_hazard = value

    @property
    def lambda_unc(self) -> float:
        return self.scoring.lambda_unc

    @lambda_unc.setter
    def lambda_unc(self, value: float) -> None:
        self.scoring.lambda_unc = value

    @property
    def lambda_cost(self) -> float:
        return self.scoring.lambda_cost

    @lambda_cost.setter
    def lambda_cost(self, value: float) -> None:
        self.scoring.lambda_cost = value

    @property
    def lambda_stale(self) -> float:
        return self.scoring.lambda_stale

    @lambda_stale.setter
    def lambda_stale(self, value: float) -> None:
        self.scoring.lambda_stale = value

    @property
    def meta_learning_range(self) -> float:
        return self.scoring.meta_learning_range

    @meta_learning_range.setter
    def meta_learning_range(self, value: float) -> None:
        self.scoring.meta_learning_range = value

    @property
    def rih_threshold(self) -> float:
        return self.assignment.rih_threshold

    @rih_threshold.setter
    def rih_threshold(self, value: float) -> None:
        self.assignment.rih_threshold = value

    @property
    def hybrid_mode(self) -> bool:
        return self.assignment.hybrid_mode

    @hybrid_mode.setter
    def hybrid_mode(self, value: bool) -> None:
        self.assignment.hybrid_mode = value

    @property
    def max_cost_ms(self) -> float:
        return self.selection.max_cost_ms

    @max_cost_ms.setter
    def max_cost_ms(self, value: float) -> None:
        self.selection.max_cost_ms = value

    @property
    def max_features(self) -> int:
        return self.selection.max_features

    @max_features.setter
    def max_features(self, value: int) -> None:
        self.selection.max_features = value

    @property
    def max_correlation(self) -> float:
        return self.selection.max_correlation

    @max_correlation.setter
    def max_correlation(self, value: float) -> None:
        self.selection.max_correlation = value

    @property
    def stability_resamples(self) -> int:
        return self.selection.stability_resamples

    @stability_resamples.setter
    def stability_resamples(self, value: int) -> None:
        self.selection.stability_resamples = value

    @property
    def fdr_q(self) -> float:
        return self.selection.fdr_q

    @fdr_q.setter
    def fdr_q(self, value: float) -> None:
        self.selection.fdr_q = value

    @property
    def min_conditional_ic(self) -> float:
        return self.selection.min_conditional_ic

    @min_conditional_ic.setter
    def min_conditional_ic(self, value: float) -> None:
        self.selection.min_conditional_ic = value

    @property
    def min_samples_for_correlation(self) -> int:
        return self.selection.min_samples_for_correlation

    @min_samples_for_correlation.setter
    def min_samples_for_correlation(self, value: int) -> None:
        self.selection.min_samples_for_correlation = value

    @property
    def embargo_minutes(self) -> int:
        return self.evaluation.embargo_minutes

    @embargo_minutes.setter
    def embargo_minutes(self, value: int) -> None:
        self.evaluation.embargo_minutes = value

    @property
    def walk_forward_folds(self) -> int:
        return self.evaluation.walk_forward_folds

    @walk_forward_folds.setter
    def walk_forward_folds(self, value: int) -> None:
        self.evaluation.walk_forward_folds = value

    @property
    def spa_test(self) -> bool:
        return self.evaluation.spa_test

    @spa_test.setter
    def spa_test(self, value: bool) -> None:
        self.evaluation.spa_test = value

    @property
    def adaptive_penalties(self) -> bool:
        return self.monitoring.adaptive_penalties

    @adaptive_penalties.setter
    def adaptive_penalties(self, value: bool) -> None:
        self.monitoring.adaptive_penalties = value

    @property
    def dashboard_enabled(self) -> bool:
        return self.monitoring.dashboard_enabled

    @dashboard_enabled.setter
    def dashboard_enabled(self, value: bool) -> None:
        self.monitoring.dashboard_enabled = value
