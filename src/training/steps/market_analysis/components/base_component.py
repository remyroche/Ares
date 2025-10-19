"""
Base component class for market analysis pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from src.training.common.component_result import ComponentError, ComponentResult
from src.training.common.artifact_persistence import SaveReport
from src.utils.logger import system_logger
from .artifact_manager import ArtifactManager

# Import ModularComponent for enhanced functionality - REMOVED
MODULAR_COMPONENT_AVAILABLE = False

@dataclass
class ComponentConfig:
    """Base configuration for pipeline components."""
    symbol: str = "ETHUSDT"
    exchange: str = "binance"
    timeframe: str = "5m"
    data_dir: str = "historical_data"
    output_dir: str = "data_cache"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    fast_mode: bool = False
    custom_params: Dict[str, Any] = None

    # Clustering-specific attributes
    min_regime_persistence: float = 0.1
    max_feature_noise_ratio: float = 0.3
    min_temporal_stability: float = 0.5
    regime_search_min: int = 2
    regime_search_max: int = 20
    n_regimes: Optional[int] = 8
    algorithm_type: Optional[str] = "regime_clustering"
    economic_weight: float = 0.3
    volatility_regime_weight: float = 0.25
    volume_regime_weight: float = 0.25
    structural_trend_weight: float = 0.2

    # Feature selection patterns
    signal_like_patterns: List[str] = None

    # Feature category caps for limiting features per category
    feature_category_caps: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}
        if self.signal_like_patterns is None:
            self.signal_like_patterns = [
                r"signal",
                r"entry",
                r"exit",
                r"crossover",
                r"trade",
            ]

class BaseMarketAnalysisComponent(ModularComponent if MODULAR_COMPONENT_AVAILABLE else ABC):
    """
    Base class for market analysis pipeline components.

    Provides common functionality and enforces consistent interface.
    Now inherits from ModularComponent for enhanced features.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the component with configuration."""
        self.config = config or ComponentConfig()
        
        if MODULAR_COMPONENT_AVAILABLE:
            # Convert ComponentConfig to dict for ModularComponent
            config_dict = {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'data_dir': self.config.data_dir,
                'output_dir': self.config.output_dir,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'force_rerun': self.config.force_rerun,
                'validation_enabled': self.config.validation_enabled,
                'monitoring_enabled': self.config.monitoring_enabled,
                'fast_mode': self.config.fast_mode,
                'custom_params': self.config.custom_params or {},
                'min_regime_persistence': self.config.min_regime_persistence,
                'max_feature_noise_ratio': self.config.max_feature_noise_ratio,
                'min_temporal_stability': self.config.min_temporal_stability,
                'regime_search_min': self.config.regime_search_min,
                'regime_search_max': self.config.regime_search_max,
                'n_regimes': self.config.n_regimes,
                'algorithm_type': self.config.algorithm_type,
                'economic_weight': self.config.economic_weight,
                'volatility_regime_weight': self.config.volatility_regime_weight,
                'volume_regime_weight': self.config.volume_regime_weight,
                'structural_trend_weight': self.config.structural_trend_weight,
                'signal_like_patterns': self.config.signal_like_patterns or [],
                'feature_category_caps': self.config.feature_category_caps or {}
            }
            super().__init__(
                name=self.__class__.__name__,
                config=config_dict,
                logger=system_logger.getChild(self.__class__.__name__)
            )
        else:
            # Fallback to original behavior
            self.logger = system_logger.getChild(self.__class__.__name__)
        
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Initialize artifact manager
        self.artifact_manager = ArtifactManager(
            base_dir="artifacts",
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe
        )

    async def save_artifacts(self, artifacts: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> SaveReport:
        """
        Save artifacts using the artifact manager.

        Args:
            artifacts: Dictionary of artifacts to save
            metadata: Optional metadata to include

        Returns:
            Dictionary mapping artifact names to file paths
        """
        component_name = self.__class__.__name__
        return await self.artifact_manager.save_artifacts(component_name, artifacts, metadata)

    async def load_artifacts_from_previous_stage(self, previous_component_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """
        Load artifacts from a previous pipeline stage.

        Args:
            previous_component_name: Name of the previous component
            artifact_names: List of artifact names to load

        Returns:
            Dictionary of loaded artifacts
        """
        return await self.artifact_manager.load_artifacts_from_previous_stage(previous_component_name, artifact_names)

    def load_artifacts_from_latest_session(self, component_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """
        Load artifacts from the most recent session.

        Args:
            component_name: Name of the component
            artifact_names: List of artifact names to load

        Returns:
            Dictionary of loaded artifacts
        """
        return self.artifact_manager.load_artifacts_from_latest_session(component_name, artifact_names)

    def validate_artifacts(self, required_artifacts: List[str]) -> bool:
        """
        Validate that all required artifacts exist and are non-empty.

        Args:
            required_artifacts: List of required artifact names

        Returns:
            True if all artifacts are valid
        """
        component_name = self.__class__.__name__
        return self.artifact_manager.validate_artifacts(component_name, required_artifacts)

    @abstractmethod
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the component logic.

        Args:
            data: Input data for the component
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with execution results
        """
        pass

    @abstractmethod
    def get_required_artifacts(self) -> list[str]:
        """
        Get list of required artifacts this component must produce.

        Returns:
            List of artifact names that must be present for success
        """
        pass

    def validate_artifacts(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate that all required artifacts are present and non-empty.

        Args:
            artifacts: Dictionary of artifacts to validate

        Returns:
            True if all required artifacts are present and valid
        """
        required_artifacts = self.get_required_artifacts()

        for artifact_name in required_artifacts:
            if artifact_name not in artifacts:
                self.logger.error(f"Missing required artifact: {artifact_name}")
                return False

            artifact_value = artifacts[artifact_name]

            # Check for empty values
            if artifact_value is None:
                self.logger.error(f"Required artifact {artifact_name} is None")
                return False
            if isinstance(artifact_value, (list, dict)) and len(artifact_value) == 0:
                self.logger.error(f"Required artifact {artifact_name} is empty")
                return False
            if isinstance(artifact_value, str) and artifact_value.strip() == "":
                self.logger.error(f"Required artifact {artifact_name} is empty string")
                return False

        return True

    def _start_execution(self):
        """Mark the start of execution."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.__class__.__name__} execution")

    def _end_execution(self) -> float:
        """Mark the end of execution and return duration."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0.0
        self.logger.info(f"Completed {self.__class__.__name__} execution in {duration:.2f}s")
        return duration

    async def save_artifacts(self, artifacts: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> SaveReport:
        """
        Save artifacts using the centralized artifact manager.

        Args:
            artifacts: Dictionary of artifacts to save
            metadata: Optional metadata to include

        Returns:
            SaveReport describing persisted artifacts

        Raises:
            Exception: If artifact saving fails
        """
        component_name = self.__class__.__name__.replace('Component', '').lower()
        return await self.artifact_manager.save_artifacts(component_name, artifacts, metadata)

    async def _execute_with_timing(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the component with timing and error handling.

        Args:
            data: Input data for the component
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with execution results
        """
        self._start_execution()

        try:
            result = await self.execute(data, pipeline_state)

            # Validate artifacts if execution was successful
            if result.success and not self.validate_artifacts(result.artifacts):
                self.logger.error("Component execution succeeded but produced invalid artifacts")
                # Clean up any partial artifacts
                component_name = self.__class__.__name__.replace('Component', '').lower()
                self.artifact_manager.cleanup_failed_artifacts(component_name)
                return ComponentResult(
                    success=False,
                    artifacts=result.artifacts,
                    error=ComponentError("Invalid artifacts produced - missing required artifacts"),
                    warnings=["Invalid artifacts produced - missing required artifacts"],
                    execution_time=self._end_execution(),
                    metadata=result.metadata,
                    metrics=result.metrics,
                )

            # Save artifacts if execution was successful
            if result.success and result.artifacts:
                try:
                    save_report = await self.save_artifacts(result.artifacts, result.metadata)
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata['artifact_save_report'] = asdict(save_report)
                    self.logger.info(
                        f"✅ Artifacts saved successfully: {list(save_report.paths.keys())} (correlation_id={save_report.correlation_id})"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Failed to save artifacts: {e}")
                    # Clean up any partial artifacts
                    component_name = self.__class__.__name__.replace('Component', '').lower()
                    self.artifact_manager.cleanup_failed_artifacts(component_name)
                    warning_message = f"Artifact saving failed: {e}"
                    return ComponentResult(
                        success=False,
                        artifacts=result.artifacts,
                        error=e,
                        warnings=[warning_message],
                        execution_time=self._end_execution(),
                        metadata=result.metadata,
                        metrics=result.metrics,
                    )

            # Update execution time
            result.execution_time = self._end_execution()
            return result

        except Exception as e:
            self.logger.error(f"Component execution failed: {e}")
            # Clean up any partial artifacts
            component_name = self.__class__.__name__.replace('Component', '').lower()
            self.artifact_manager.cleanup_failed_artifacts(component_name)
            warning_message = f"Component execution failed: {e}"
            return ComponentResult(
                success=False,
                artifacts={},
                error=e,
                warnings=[warning_message],
                execution_time=self._end_execution(),
                metrics={},
            )

    # Required abstract methods from ModularComponent (if available)
    if MODULAR_COMPONENT_AVAILABLE:
        def _initialize_resources(self) -> bool:
            """Initialize component-specific resources."""
            try:
                # Set initial state
                self.set_state('initialized_at', datetime.now().isoformat())
                self.set_state('execution_count', 0)
                return True
            except Exception as e:
                self.logger.error(f"Resource initialization failed: {e}")
                return False
        
        def _cleanup_resources(self) -> None:
            """Cleanup component-specific resources."""
            self.set_state('cleaned_up_at', datetime.now().isoformat())
            self.set_state('execution_count', 0)
        
        def _process_data(self, data: Any, **kwargs) -> Any:
            """Process data with component logic."""
            # Increment execution count
            count = self.get_state('execution_count', 0)
            self.set_state('execution_count', count + 1)
            
            # Basic processing - return data as-is for now
            # The actual processing is done in the execute method
            return data
        
        def _get_validation_rules(self) -> Dict[str, Any]:
            """Get validation rules for this component."""
            return {
                'min_size': 100,
                'max_size': 1000000,
                'required_attributes': ['open', 'high', 'low', 'close', 'volume'],
                'data_types': ['pandas.DataFrame'],
                'max_nan_ratio': 0.1,
                'min_unique_values': 2
            }
        
        def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
            """Validate data with component-specific rules."""
            errors = []
            warnings = []
            metadata = {}
            
            if hasattr(data, 'shape'):  # DataFrame-like object
                # Check data size
                if len(data) < 100:
                    warnings.append("Data size is small (< 100 rows)")
                
                # Check for NaN values
                if hasattr(data, 'isnull'):
                    nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                    if nan_ratio > 0.1:
                        warnings.append(f"High NaN ratio: {nan_ratio:.2%}")
                    
                    metadata['data_shape'] = data.shape
                    metadata['nan_ratio'] = nan_ratio
                    metadata['columns'] = list(data.columns)
            
            return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
