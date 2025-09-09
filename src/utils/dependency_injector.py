from __future__ import annotations
"""Dependency injection façade for step modules.

This module provides a stable public import path for the UtilityDependencyInjector that
lives inside the (currently very large) step03_5 implementation.  Downstream code should
import the injector from here instead of reaching deep into step files.

Eventually the full implementation will be moved here.  For now we re-export the
existing class to avoid large immediate code moves while still giving us a single point
of reference.
"""

try:
    from src.training.steps.market_analysis.hmm_clustering.step03_5_final_regime_clustering import (
        UtilityDependencyInjector as _OldUtilityDependencyInjector,
    )
except ImportError as exc:  # pragma: no cover – should only fail in specialised build envs
    raise ImportError(
        "Could not import UtilityDependencyInjector from step03_5 module. "
        "Make sure that module is importable before using dependency_injector."
    ) from exc


class UtilityDependencyInjector(_OldUtilityDependencyInjector):
    """Enhanced façade that tweaks logging verbosity and platform-specific injections.

    The long-term goal is to move the full implementation here; meanwhile we inherit from
    the original class but patch a few behaviours:
    1. All internal *info-level* log messages are downgraded to DEBUG to reduce noise.
    2. M1-specific utility injections are skipped on non-Apple-Silicon platforms.
    """

    # --- logging tweaks -----------------------------------------------------
    def __init__(self, config: dict[str, Any], logger: "logging.Logger") -> None:  # type: ignore[name-defined]
        # Import late to avoid unnecessary cost if the injector is never used.
        import logging  # noqa: WPS433 (allow stdlib import inside function)
        super().__init__(config, logger)
        # Downgrade noisy info-level calls by aliasing .info to .debug
        self.logger.info = self.logger.debug  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Platform helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _is_m1_platform() -> bool:
        """Return *True* if running on macOS + Apple-Silicon (arm64)."""
        import platform  # noqa: WPS433
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    # -----------------------------------------------------------------------
    # Overrides for M1-specific injections to add platform guard.
    # -----------------------------------------------------------------------
    def _inject_m1_gpu_utils(self) -> None:  # noqa: D401
        if not self._is_m1_platform():
            # Mark as skipped / not applicable.
            self.logger.debug("Skipping M1 GPU utilities; non-Apple-Silicon platform detected.")
            self._initialization_status["m1_gpu_utils"] = False
            return
        super()._inject_m1_gpu_utils()

    def _inject_m1_memory_optimizer(self) -> None:  # noqa: D401
        if not self._is_m1_platform():
            self.logger.debug("Skipping M1 Memory optimizer utilities; non-Apple-Silicon platform detected.")
            self._initialization_status["m1_memory_optimizer"] = False
            return
        super()._inject_m1_memory_optimizer()

    def _inject_m1_cpu_optimizer(self) -> None:  # noqa: D401
        if not self._is_m1_platform():
            self.logger.debug("Skipping M1 CPU optimizer utilities; non-Apple-Silicon platform detected.")
            self._initialization_status["m1_cpu_optimizer"] = False
            return
        super()._inject_m1_cpu_optimizer()