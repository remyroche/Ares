"""Compatibility wrapper around the canonical unified hardware manager."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.hardware.unified_hardware_manager import (
    HardwareConfig,
    OptimizationLevel,
    PerformanceMetrics,
    UnifiedHardwareManager as CanonicalUnifiedHardwareManager,
    WorkloadType,
    get_system_status as _get_system_status,
    get_unified_hardware_manager as _get_unified_hardware_manager,
    optimize_for_workload as _optimize_for_workload,
    shutdown_hardware_manager as _shutdown_hardware_manager,
)

try:  # Optional import used only for typing and logging context.
    from ..unified_architecture_config import ArchitectureType
except ImportError:  # pragma: no cover - legacy environments may lack this module.
    ArchitectureType = Any  # type: ignore


class HardwareType(Enum):
    """High-level hardware resource categories."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class HardwareMetrics:
    """Light-weight metrics container maintained for backwards compatibility."""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    temperature: float = 0.0
    power_consumption: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0

    @classmethod
    def from_performance_metrics(cls, metrics: PerformanceMetrics) -> "HardwareMetrics":
        """Create a :class:`HardwareMetrics` instance from the canonical dataclass."""

        return cls(
            timestamp=metrics.timestamp,
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            gpu_usage=getattr(metrics, "gpu_usage", 0.0),
            gpu_memory_usage=getattr(metrics, "gpu_memory_usage", 0.0),
            temperature=getattr(metrics, "temperature", 0.0),
            power_consumption=getattr(metrics, "power_consumption", 0.0),
            throughput=getattr(metrics, "throughput", 0.0),
            latency=getattr(metrics, "latency", 0.0),
        )


class UnifiedHardwareManager(CanonicalUnifiedHardwareManager):
    """Shim that forwards to :class:`src.utils.hardware.unified_hardware_manager`."""

    def __init__(
        self,
        architecture_type: Optional[ArchitectureType] = None,
        config: Optional[HardwareConfig] = None,
    ) -> None:
        self.architecture_type = architecture_type
        super().__init__(config=config)

    def describe(self) -> Dict[str, Any]:
        """Return the canonical system status enriched with architecture context."""

        status = self.get_system_status()
        status.setdefault(
            "architecture_type",
            getattr(self.architecture_type, "value", self.architecture_type),
        )
        return status


def create_hardware_manager(
    architecture_type: Optional[ArchitectureType] = None,
    *,
    config: Optional[HardwareConfig] = None,
    **config_overrides: Any,
) -> UnifiedHardwareManager:
    """Instantiate a hardware manager that delegates to the canonical manager."""

    effective_config = config or HardwareConfig(**config_overrides)
    return UnifiedHardwareManager(architecture_type=architecture_type, config=effective_config)


def create_basic_hardware_manager(
    architecture_type: Optional[ArchitectureType] = None,
) -> UnifiedHardwareManager:
    """Create a conservative hardware manager configuration."""

    config = HardwareConfig(
        cpu_optimization_level=OptimizationLevel.MINIMAL,
        gpu_optimization_level=OptimizationLevel.MINIMAL,
        memory_optimization_level=OptimizationLevel.MINIMAL,
        enable_adaptive_optimization=False,
        enable_performance_monitoring=False,
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def create_aggressive_hardware_manager(
    architecture_type: Optional[ArchitectureType] = None,
) -> UnifiedHardwareManager:
    """Create an aggressive hardware manager tuned for peak performance."""

    config = HardwareConfig(
        cpu_optimization_level=OptimizationLevel.MAXIMUM,
        gpu_optimization_level=OptimizationLevel.MAXIMUM,
        memory_optimization_level=OptimizationLevel.MAXIMUM,
        enable_adaptive_optimization=True,
        enable_performance_monitoring=True,
        monitoring_interval=1.0,
    )
    return UnifiedHardwareManager(architecture_type=architecture_type, config=config)


def get_hardware_manager(
    config: Optional[HardwareConfig] = None,
    *,
    conservative_mode: bool = False,
) -> CanonicalUnifiedHardwareManager:
    """Return the process-wide canonical hardware manager instance."""

    return _get_unified_hardware_manager(config=config, conservative_mode=conservative_mode)


def optimize_system_performance(
    workload_type: WorkloadType,
    optimization_level: Optional[OptimizationLevel] = None,
) -> bool:
    """Proxy to :func:`src.utils.hardware.unified_hardware_manager.optimize_for_workload`."""

    return _optimize_for_workload(workload_type, optimization_level)


def get_hardware_status() -> Dict[str, Any]:
    """Expose the canonical system status helper under the legacy name."""

    return _get_system_status()


def shutdown_hardware_manager() -> None:
    """Shutdown the shared hardware manager instance."""

    _shutdown_hardware_manager()


__all__ = [
    "UnifiedHardwareManager",
    "HardwareConfig",
    "HardwareType",
    "HardwareMetrics",
    "WorkloadType",
    "OptimizationLevel",
    "PerformanceMetrics",
    "create_hardware_manager",
    "create_basic_hardware_manager",
    "create_aggressive_hardware_manager",
    "get_hardware_manager",
    "optimize_system_performance",
    "get_hardware_status",
    "shutdown_hardware_manager",
]
