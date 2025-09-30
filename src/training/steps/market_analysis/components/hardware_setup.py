"""Hardware setup utilities for NAS-TAS clustering components."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.tprint import tprint

# Import matrix operations and hardware utilities
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_batch_matrix_processor,
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as error:  # pragma: no cover - import guard
    MATRIX_OPERATIONS_AVAILABLE = False
    tprint(f"Matrix operations not available: {error}", "WARNING")

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        get_adaptive_optimization_engine,
        optimize_for_workload,
        optimize_for_workload_adaptive,
        optimize_dataframe_advanced,
        record_performance_adaptive,
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    tprint("✅ Hardware optimization utilities imported successfully", "SUCCESS")
except ImportError as error:  # pragma: no cover - import guard
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    tprint(f"Hardware optimization not available: {error}", "WARNING")

try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_optimizer,
        get_m1_gpu_memory_manager,
        get_m1_gpu_performance_monitor,
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer,
        get_m1_memory_pool_manager,
        get_m1_memory_monitor,
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer,
        get_m1_cpu_performance_monitor,
        get_m1_cpu_scheduler,
    )
    M1_HARDWARE_AVAILABLE = True
    tprint("✅ M1-specific hardware utilities imported successfully", "SUCCESS")
except ImportError as error:  # pragma: no cover - import guard
    M1_HARDWARE_AVAILABLE = False
    tprint(f"M1 hardware utilities not available: {error}", "WARNING")

    # Provide safe fallbacks to avoid AttributeErrors when not available
    get_m1_gpu_optimizer = lambda: None  # type: ignore
    get_m1_gpu_memory_manager = lambda: None  # type: ignore
    get_m1_gpu_performance_monitor = lambda: None  # type: ignore
    get_m1_memory_optimizer = lambda: None  # type: ignore
    get_m1_memory_pool_manager = lambda: None  # type: ignore
    get_m1_memory_monitor = lambda: None  # type: ignore
    get_m1_cpu_optimizer = lambda: None  # type: ignore
    get_m1_cpu_performance_monitor = lambda: None  # type: ignore
    get_m1_cpu_scheduler = lambda: None  # type: ignore


@dataclass
class HardwareResources:
    """Collection of hardware and matrix resources used by clustering components."""

    matrix_ops: Optional[Any] = None
    vectorized_core: Optional[Any] = None
    batch_processor: Optional[Any] = None
    hardware_manager: Optional[Any] = None
    m1_gpu_optimizer: Optional[Any] = None
    m1_memory_optimizer: Optional[Any] = None
    m1_cpu_optimizer: Optional[Any] = None


class HardwareSetup:
    """Helper responsible for initializing hardware and matrix resources."""

    def initialize(self) -> HardwareResources:
        """Initialize available hardware optimizations and matrix operations."""
        tprint("🔧 Initializing hardware optimization systems...", "INFO")
        hardware_resources = self._initialize_hardware_optimization()

        tprint("📊 Initializing matrix operations with hardware optimization...", "INFO")
        matrix_resources = self._initialize_matrix_operations()

        combined_resources: Dict[str, Optional[Any]] = {
            **matrix_resources,
            **hardware_resources,
        }
        return HardwareResources(**combined_resources)

    def _initialize_hardware_optimization(self) -> Dict[str, Optional[Any]]:
        resources: Dict[str, Optional[Any]] = {
            "hardware_manager": None,
            "m1_gpu_optimizer": None,
            "m1_memory_optimizer": None,
            "m1_cpu_optimizer": None,
        }

        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                resources["hardware_manager"] = get_unified_hardware_manager()
                tprint("✅ Hardware optimization initialized successfully", "SUCCESS")
            except Exception as error:  # pragma: no cover - defensive logging
                tprint(f"❌ Failed to initialize unified hardware manager: {error}", "ERROR")
        else:
            tprint(
                "⚠️  Hardware optimization utilities not available, using fallback",
                "WARNING",
            )

        if M1_HARDWARE_AVAILABLE:
            try:
                tprint("  🎮 Initializing M1 GPU optimizer...", "INFO")
                resources["m1_gpu_optimizer"] = get_m1_gpu_optimizer()
                if resources["m1_gpu_optimizer"]:
                    tprint("  ✅ M1 GPU optimizer initialized", "SUCCESS")

                tprint("  💾 Initializing M1 memory optimizer...", "INFO")
                resources["m1_memory_optimizer"] = get_m1_memory_optimizer()
                if resources["m1_memory_optimizer"]:
                    tprint("  ✅ M1 memory optimizer initialized", "SUCCESS")

                tprint("  🖥️  Initializing M1 CPU optimizer...", "INFO")
                resources["m1_cpu_optimizer"] = get_m1_cpu_optimizer()
                if resources["m1_cpu_optimizer"]:
                    tprint("  ✅ M1 CPU optimizer initialized", "SUCCESS")

                tprint("✅ M1 hardware optimization systems initialized", "SUCCESS")
            except Exception as error:  # pragma: no cover - defensive logging
                tprint(f"❌ M1 hardware optimization initialization failed: {error}", "ERROR")
        else:
            tprint(
                "⚠️  M1 hardware utilities not available, using fallback",
                "WARNING",
            )

        return resources

    def _initialize_matrix_operations(self) -> Dict[str, Optional[Any]]:
        resources: Dict[str, Optional[Any]] = {
            "matrix_ops": None,
            "vectorized_core": None,
            "batch_processor": None,
        }

        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                tprint("  🔄 Initializing unified matrix operations...", "INFO")
                resources["matrix_ops"] = get_unified_matrix_operations()
                if resources["matrix_ops"]:
                    tprint("  ✅ Unified matrix operations initialized", "SUCCESS")

                tprint("  ⚡ Initializing vectorized processing core...", "INFO")
                resources["vectorized_core"] = get_vectorized_processing_core()
                if resources["vectorized_core"]:
                    tprint("  ✅ Vectorized processing core initialized", "SUCCESS")

                tprint("  📦 Initializing batch matrix processor...", "INFO")
                resources["batch_processor"] = get_batch_matrix_processor()
                if resources["batch_processor"]:
                    tprint("  ✅ Batch matrix processor initialized", "SUCCESS")

                tprint(
                    "✅ Matrix operations initialized with hardware optimization",
                    "SUCCESS",
                )
            except Exception as error:  # pragma: no cover - defensive logging
                tprint(f"❌ Matrix operations initialization failed: {error}", "ERROR")
        else:
            tprint("⚠️  Matrix operations not available, using fallback", "WARNING")

        return resources
