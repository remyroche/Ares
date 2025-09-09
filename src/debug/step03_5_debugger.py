import logging
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional


class Step03_5Debugger:
    """Comprehensive debugger utility for Step 3.5 (FinalRegimeClusteringStep).

    This tool helps diagnose common issues during the initialization phase of
    `FinalRegimeClusteringStep`, verifies the presence of critical utilities and
    optimisation components, and runs lightweight validation checks. It is
    designed to be **non-destructive** and can be safely executed in any
    environment (CI, local, production) because it never performs write or
    delete operations on the source data or models.
    """

    #: Fully-qualified dotted path to the class we want to debug
    _STEP_CLASS_PATH = (
        "src.training.steps.market_analysis.hmm_clustering."
        "step03_5_final_regime_clustering.FinalRegimeClusteringStep"
    )

    #: Methods we expect to exist on the `FinalRegimeClusteringStep` class
    _EXPECTED_METHODS: List[str] = [
        "_log_utility_integration_status",
        "_perform_comprehensive_utility_operations",
        "_load_data_with_comprehensive_utilities",
        "_prepare_features_with_comprehensive_utilities",
        "_perform_hmm_regime_discovery_with_utilities",
        "_perform_final_clustering_with_utilities",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, logger: Optional[logging.Logger] = None) -> None:
        #: Configuration forwarded to the step. Defaults to an empty dict.
        self.config = config or {}

        #: Dedicated logger for the debugger
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        self.logger.debug("Initialising Step03_5Debugger…")

        # Import step03_5 class dynamically so that users can run the debugger
        # without having the entire training dependency tree installed.
        self._step_cls = self._import_step_class()

    # ---------------------------------------------------------------------
    # Public high-level API
    # ---------------------------------------------------------------------
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run all available diagnostics and return a structured report."""
        self.logger.info("🚀 Starting full diagnostics for Step 3.5 …")

        report: Dict[str, Any] = {
            "import_check": self._diagnose_imports(),
            "method_check": self._diagnose_methods(),
            "initialisation_check": self._diagnose_initialisation(),
        }

        self.logger.info("🎉 Diagnostics complete")
        return report

    # ------------------------------------------------------------------
    # Individual diagnostic helpers
    # ------------------------------------------------------------------
    def _diagnose_imports(self) -> Dict[str, Any]:
        """Verify that the step class can be imported."""
        successful = self._step_cls is not None
        return {
            "success": successful,
            "step_class": self._step_cls,
            "error": None if successful else "Failed to import step class",
        }

    def _diagnose_methods(self) -> Dict[str, Any]:
        """Check that all expected methods are present on the step class."""
        if self._step_cls is None:
            return {
                "success": False,
                "missing_methods": self._EXPECTED_METHODS,
                "present_methods": [],
            }

        present = []
        missing = []
        for method in self._EXPECTED_METHODS:
            if hasattr(self._step_cls, method):
                present.append(method)
            else:
                missing.append(method)

        return {
            "success": len(missing) == 0,
            "missing_methods": missing,
            "present_methods": present,
        }

    def _diagnose_initialisation(self) -> Dict[str, Any]:
        """Attempt to instantiate the step with a lightweight config.

        We intercept any exception and include the traceback in the report
        without letting it propagate. The instantiation is run in isolation and
        should therefore be safe.
        """
        if self._step_cls is None:
            return {
                "success": False,
                "error": "Cannot instantiate – import failed",
            }

        try:
            instance = self._step_cls(self.config)
            return {
                "success": True,
                "instance": instance,
            }
        except Exception as exc:  # pylint: disable=broad-except
            import traceback

            tb = traceback.format_exc()
            self.logger.error("❌ Initialisation failed: %s", exc)
            self.logger.debug(tb)
            return {
                "success": False,
                "error": str(exc),
                "traceback": tb,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _import_step_class(self):
        """Dynamically import the `FinalRegimeClusteringStep` class."""
        module_path, _, class_name = self._STEP_CLASS_PATH.rpartition(".")
        try:
            module: ModuleType = importlib.import_module(module_path)
            step_cls = getattr(module, class_name)
            self.logger.debug("Successfully imported %s", self._STEP_CLASS_PATH)
            return step_cls
        except (ImportError, AttributeError) as exc:
            self.logger.error("❌ Import error for %s: %s", self._STEP_CLASS_PATH, exc)
            return None


# -------------------------------------------------------------------------
# Convenience CLI usage: `python -m src.debug.step03_5_debugger <config_path>`
# -------------------------------------------------------------------------

def _load_config_from_file(path: Path | str) -> Dict[str, Any]:
    """Load configuration from JSON/YAML file or return empty dict on failure."""
    import json
    try:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)
        with file_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:  # pylint: disable=broad-except
        return {}


def main() -> None:  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Debug Step 3.5 – FinalRegimeClusteringStep")
    parser.add_argument("config", nargs="?", help="Path to JSON config file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = _load_config_from_file(args.config) if args.config else {}

    debugger = Step03_5Debugger(cfg)
    report = debugger.run_full_diagnostics()

    print("\n===== Step03_5 Diagnostics Report =====")
    import pprint

    pprint.pprint(report, sort_dicts=False)


if __name__ == "__main__":  # pragma: no cover
    main()