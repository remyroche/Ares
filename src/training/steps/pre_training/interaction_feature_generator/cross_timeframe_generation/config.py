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

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint_debug
    TPRINT_AVAILABLE = True
    tprint_debug("tprint_debug imported successfully for cross-timeframe configuration")
except ImportError:
    TPRINT_AVAILABLE = False

    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)

    tprint_debug(
        "Using fallback tprint_debug implementation for cross-timeframe configuration"
    )


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
    bootstrap_block_size: Optional[int] = None
    bootstrap_resamples: int = 1000
    bootstrap_confidence_level: float = 0.95
    bootstrap_random_seed: Optional[int] = None
    hac_max_lag: Optional[int] = None


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
        tprint_debug(
            "Initializing PipelineConfig",
            session=session,
            probe=probe,
            optimization=optimization,
            regime=regime,
            scoring=scoring,
            assignment=assignment,
            selection=selection,
            evaluation=evaluation,
            monitoring=monitoring,
            overrides=overrides,
        )
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
            tprint_debug("Applied override", key=key, value=value)

    # ------------------------------------------------------------------
    # Convenience helpers mirroring the legacy flat PipelineConfig API.
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return default

    @property
    def base_timeframe_minutes(self) -> int:
        value = self.session.base_timeframe_minutes
        tprint_debug("Accessed base_timeframe_minutes", value=value)
        return value

    @base_timeframe_minutes.setter
    def base_timeframe_minutes(self, value: int) -> None:
        tprint_debug("Updated base_timeframe_minutes", value=value)
        self.session.base_timeframe_minutes = value

    @property
    def session_start_hour(self) -> int:
        value = self.session.session_start_hour
        tprint_debug("Accessed session_start_hour", value=value)
        return value

    @session_start_hour.setter
    def session_start_hour(self, value: int) -> None:
        tprint_debug("Updated session_start_hour", value=value)
        self.session.session_start_hour = value

    @property
    def session_end_hour(self) -> int:
        value = self.session.session_end_hour
        tprint_debug("Accessed session_end_hour", value=value)
        return value

    @session_end_hour.setter
    def session_end_hour(self, value: int) -> None:
        tprint_debug("Updated session_end_hour", value=value)
        self.session.session_end_hour = value

    @property
    def dst_handling(self) -> bool:
        value = self.session.dst_handling
        tprint_debug("Accessed dst_handling", value=value)
        return value

    @dst_handling.setter
    def dst_handling(self, value: bool) -> None:
        tprint_debug("Updated dst_handling", value=value)
        self.session.dst_handling = value

    @property
    def coarse_grid_min(self) -> int:
        value = self.probe.coarse_grid_min
        tprint_debug("Accessed coarse_grid_min", value=value)
        return value

    @coarse_grid_min.setter
    def coarse_grid_min(self, value: int) -> None:
        tprint_debug("Updated coarse_grid_min", value=value)
        self.probe.coarse_grid_min = value

    @property
    def coarse_grid_max(self) -> int:
        value = self.probe.coarse_grid_max
        tprint_debug("Accessed coarse_grid_max", value=value)
        return value

    @coarse_grid_max.setter
    def coarse_grid_max(self, value: int) -> None:
        tprint_debug("Updated coarse_grid_max", value=value)
        self.probe.coarse_grid_max = value

    @property
    def adaptive_refinement_threshold(self) -> float:
        value = self.probe.adaptive_refinement_threshold
        tprint_debug("Accessed adaptive_refinement_threshold", value=value)
        return value

    @adaptive_refinement_threshold.setter
    def adaptive_refinement_threshold(self, value: float) -> None:
        tprint_debug("Updated adaptive_refinement_threshold", value=value)
        self.probe.adaptive_refinement_threshold = value

    @property
    def local_grid_factor(self) -> float:
        value = self.optimization.local_grid_factor
        tprint_debug("Accessed local_grid_factor", value=value)
        return value

    @local_grid_factor.setter
    def local_grid_factor(self, value: float) -> None:
        tprint_debug("Updated local_grid_factor", value=value)
        self.optimization.local_grid_factor = value

    @property
    def ic_surface_smoothing(self) -> str:
        value = self.optimization.ic_surface_smoothing
        tprint_debug("Accessed ic_surface_smoothing", value=value)
        return value

    @ic_surface_smoothing.setter
    def ic_surface_smoothing(self, value: str) -> None:
        tprint_debug("Updated ic_surface_smoothing", value=value)
        self.optimization.ic_surface_smoothing = value

    @property
    def change_point_method(self) -> str:
        value = self.regime.change_point_method
        tprint_debug("Accessed change_point_method", value=value)
        return value

    @change_point_method.setter
    def change_point_method(self, value: str) -> None:
        tprint_debug("Updated change_point_method", value=value)
        self.regime.change_point_method = value

    @property
    def regime_vol_quantile(self) -> float:
        value = self.regime.regime_vol_quantile
        tprint_debug("Accessed regime_vol_quantile", value=value)
        return value

    @regime_vol_quantile.setter
    def regime_vol_quantile(self, value: float) -> None:
        tprint_debug("Updated regime_vol_quantile", value=value)
        self.regime.regime_vol_quantile = value

    @property
    def bocpd_hazard(self) -> float:
        value = self.regime.bocpd_hazard
        tprint_debug("Accessed bocpd_hazard", value=value)
        return value

    @bocpd_hazard.setter
    def bocpd_hazard(self, value: float) -> None:
        tprint_debug("Updated bocpd_hazard", value=value)
        self.regime.bocpd_hazard = value

    @property
    def lambda_unc(self) -> float:
        value = self.scoring.lambda_unc
        tprint_debug("Accessed lambda_unc", value=value)
        return value

    @lambda_unc.setter
    def lambda_unc(self, value: float) -> None:
        tprint_debug("Updated lambda_unc", value=value)
        self.scoring.lambda_unc = value

    @property
    def lambda_cost(self) -> float:
        value = self.scoring.lambda_cost
        tprint_debug("Accessed lambda_cost", value=value)
        return value

    @lambda_cost.setter
    def lambda_cost(self, value: float) -> None:
        tprint_debug("Updated lambda_cost", value=value)
        self.scoring.lambda_cost = value

    @property
    def lambda_stale(self) -> float:
        value = self.scoring.lambda_stale
        tprint_debug("Accessed lambda_stale", value=value)
        return value

    @lambda_stale.setter
    def lambda_stale(self, value: float) -> None:
        tprint_debug("Updated lambda_stale", value=value)
        self.scoring.lambda_stale = value

    @property
    def meta_learning_range(self) -> float:
        value = self.scoring.meta_learning_range
        tprint_debug("Accessed meta_learning_range", value=value)
        return value

    @meta_learning_range.setter
    def meta_learning_range(self, value: float) -> None:
        tprint_debug("Updated meta_learning_range", value=value)
        self.scoring.meta_learning_range = value

    @property
    def rih_threshold(self) -> float:
        value = self.assignment.rih_threshold
        tprint_debug("Accessed rih_threshold", value=value)
        return value

    @rih_threshold.setter
    def rih_threshold(self, value: float) -> None:
        tprint_debug("Updated rih_threshold", value=value)
        self.assignment.rih_threshold = value

    @property
    def hybrid_mode(self) -> bool:
        value = self.assignment.hybrid_mode
        tprint_debug("Accessed hybrid_mode", value=value)
        return value

    @hybrid_mode.setter
    def hybrid_mode(self, value: bool) -> None:
        tprint_debug("Updated hybrid_mode", value=value)
        self.assignment.hybrid_mode = value

    @property
    def max_cost_ms(self) -> float:
        value = self.selection.max_cost_ms
        tprint_debug("Accessed max_cost_ms", value=value)
        return value

    @max_cost_ms.setter
    def max_cost_ms(self, value: float) -> None:
        tprint_debug("Updated max_cost_ms", value=value)
        self.selection.max_cost_ms = value

    @property
    def max_features(self) -> int:
        value = self.selection.max_features
        tprint_debug("Accessed max_features", value=value)
        return value

    @max_features.setter
    def max_features(self, value: int) -> None:
        tprint_debug("Updated max_features", value=value)
        self.selection.max_features = value

    @property
    def max_correlation(self) -> float:
        value = self.selection.max_correlation
        tprint_debug("Accessed max_correlation", value=value)
        return value

    @max_correlation.setter
    def max_correlation(self, value: float) -> None:
        tprint_debug("Updated max_correlation", value=value)
        self.selection.max_correlation = value

    @property
    def stability_resamples(self) -> int:
        value = self.selection.stability_resamples
        tprint_debug("Accessed stability_resamples", value=value)
        return value

    @stability_resamples.setter
    def stability_resamples(self, value: int) -> None:
        tprint_debug("Updated stability_resamples", value=value)
        self.selection.stability_resamples = value

    @property
    def fdr_q(self) -> float:
        value = self.selection.fdr_q
        tprint_debug("Accessed fdr_q", value=value)
        return value

    @fdr_q.setter
    def fdr_q(self, value: float) -> None:
        tprint_debug("Updated fdr_q", value=value)
        self.selection.fdr_q = value

    @property
    def min_conditional_ic(self) -> float:
        value = self.selection.min_conditional_ic
        tprint_debug("Accessed min_conditional_ic", value=value)
        return value

    @min_conditional_ic.setter
    def min_conditional_ic(self, value: float) -> None:
        tprint_debug("Updated min_conditional_ic", value=value)
        self.selection.min_conditional_ic = value

    @property
    def min_samples_for_correlation(self) -> int:
        value = self.selection.min_samples_for_correlation
        tprint_debug("Accessed min_samples_for_correlation", value=value)
        return value

    @min_samples_for_correlation.setter
    def min_samples_for_correlation(self, value: int) -> None:
        tprint_debug("Updated min_samples_for_correlation", value=value)
        self.selection.min_samples_for_correlation = value

    @property
    def embargo_minutes(self) -> int:
        value = self.evaluation.embargo_minutes
        tprint_debug("Accessed embargo_minutes", value=value)
        return value

    @embargo_minutes.setter
    def embargo_minutes(self, value: int) -> None:
        tprint_debug("Updated embargo_minutes", value=value)
        self.evaluation.embargo_minutes = value

    @property
    def walk_forward_folds(self) -> int:
        value = self.evaluation.walk_forward_folds
        tprint_debug("Accessed walk_forward_folds", value=value)
        return value

    @walk_forward_folds.setter
    def walk_forward_folds(self, value: int) -> None:
        tprint_debug("Updated walk_forward_folds", value=value)
        self.evaluation.walk_forward_folds = value

    @property
    def spa_test(self) -> bool:
        value = self.evaluation.spa_test
        tprint_debug("Accessed spa_test", value=value)
        return value

    @spa_test.setter
    def spa_test(self, value: bool) -> None:
        tprint_debug("Updated spa_test", value=value)
        self.evaluation.spa_test = value

    @property
    def adaptive_penalties(self) -> bool:
        value = self.monitoring.adaptive_penalties
        tprint_debug("Accessed adaptive_penalties", value=value)
        return value

    @adaptive_penalties.setter
    def adaptive_penalties(self, value: bool) -> None:
        tprint_debug("Updated adaptive_penalties", value=value)
        self.monitoring.adaptive_penalties = value

    @property
    def dashboard_enabled(self) -> bool:
        value = self.monitoring.dashboard_enabled
        tprint_debug("Accessed dashboard_enabled", value=value)
        return value

    @dashboard_enabled.setter
    def dashboard_enabled(self, value: bool) -> None:
        tprint_debug("Updated dashboard_enabled", value=value)
        self.monitoring.dashboard_enabled = value
