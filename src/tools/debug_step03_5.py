#!/usr/bin/env python3
"""
Step03_5 Debug Utilities

Tools to thoroughly debug Step 3.5 (Final Regime Clustering):
- Enable verbose DEBUG logging
- Run full step or selected sub-steps in isolation
- Export detailed function-call monitor reports
- Export performance monitor summaries
- Dump utility injection and environment diagnostics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# Ensure project root on sys.path when invoked as a script
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.logger import get_logger, log_dataframe_overview  # type: ignore
from src.utils.function_call_monitor import (
    get_function_call_monitor,
    log_function_call_summary,
)  # type: ignore
from src.utils.financial_metrics_logger import get_financial_metrics_logger  # type: ignore


from src.training.steps.market_analysis.hmm_clustering.step03_5_final_regime_clustering import (  # type: ignore
    FinalRegimeClusteringStep,
)


class Step03_5Debugger:
    """Orchestrates thorough debugging for Step 3.5 with reporting artifacts."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or (PROJECT_ROOT / "logs" / "step03_5_debug" / timestamp))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("Step03_5Debugger")

    # ---------- Public API ----------
    def enable_verbose_logging(self) -> None:
        """Elevate log levels for detailed debugging output."""
        try:
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            for handler in list(root_logger.handlers):
                handler.setLevel(logging.DEBUG)
            # Ensure our module logger is also at DEBUG
            self.logger.setLevel(logging.DEBUG)
            for handler in list(self.logger.handlers):
                handler.setLevel(logging.DEBUG)
            self.logger.info("🔧 Enabled DEBUG logging for thorough diagnostics")
        except Exception as e:
            # Avoid failing the debug run due to logging setup
            print(f"Failed to enable verbose logging: {e}")

    def export_function_call_report(self) -> Path:
        """Export detailed function call monitoring report to JSON."""
        monitor = get_function_call_monitor()
        report_path = self.output_dir / "function_calls_detailed.json"
        try:
            monitor.export_detailed_report(str(report_path))
            self.logger.info(f"📊 Function-call detailed report exported: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to export function-call report: {e}")
        return report_path

    def export_environment_snapshot(self) -> Path:
        """Write environment and runtime diagnostics to JSON."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "process": {
                "pid": os.getpid(),
                "cwd": os.getcwd(),
            },
        }
        path = self.output_dir / "environment_snapshot.json"
        try:
            path.write_text(json.dumps(snapshot, indent=2))
            self.logger.info(f"🧭 Environment snapshot exported: {path}")
        except Exception as e:
            self.logger.error(f"Failed to write environment snapshot: {e}")
        return path

    async def run_full(self, config: Dict[str, Any]) -> bool:
        """Run full Step 3.5 with instrumentation and export debug artifacts."""
        start = time.time()
        step = FinalRegimeClusteringStep(config)
        try:
            await step.initialize()
            success = await step.execute()
            await step.cleanup()
            # Export performance summary if available
            try:
                perf_summary = step.performance_monitor.get_summary() if hasattr(step, "performance_monitor") else {}
            except Exception:
                perf_summary = {}
            self._export_performance_summary(perf_summary)
            # Export utility status
            self._export_utility_status(step)
            # Write function-call report
            self.export_function_call_report()
            # Log brief monitor summary to logs
            log_function_call_summary(self.logger)
            elapsed = time.time() - start
            self.logger.info(f"✅ Full Step03_5 debug run completed in {elapsed:.2f}s | success={success}")
            return success
        except Exception as e:
            self.logger.exception(f"❌ Step03_5 debug run failed: {e}")
            # Still try to export artifacts for post-mortem
            try:
                self.export_function_call_report()
            except Exception:
                pass
            return False

    async def run_substeps(self, config: Dict[str, Any], only: str) -> bool:
        """Run selected sub-steps for targeted debugging.

        only: one of {'data','hmm','clustering','analysis','reports','save'}
        """
        start = time.time()
        step = FinalRegimeClusteringStep(config)
        await step.initialize()

        result: Dict[str, Any] = {}

        try:
            if only == "data":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                result = {"stage": "data", "success": bool(data_loaded.get("success")), "keys": list(data_loaded.keys())}
                # DataFrame overview if present
                if "data" in data_loaded and hasattr(data_loaded["data"], "head"):
                    log_dataframe_overview(step.logger, data_loaded["data"], name="loaded_data", sample_rows=3)

            elif only == "hmm":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                hmm_res = await step._perform_hmm_regime_discovery_with_utilities(data_loaded["data"])  # type: ignore[attr-defined]
                result = {"stage": "hmm", "keys": list(hmm_res.keys())}

            elif only == "clustering":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                hmm_res = await step._perform_hmm_regime_discovery_with_utilities(data_loaded["data"])  # type: ignore[attr-defined]
                clustering = await step._perform_final_clustering_with_utilities(data_loaded["data"], hmm_res)  # type: ignore[attr-defined]
                result = {"stage": "clustering", "keys": list(clustering.keys())}

            elif only == "analysis":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                hmm_res = await step._perform_hmm_regime_discovery_with_utilities(data_loaded["data"])  # type: ignore[attr-defined]
                clustering = await step._perform_final_clustering_with_utilities(data_loaded["data"], hmm_res)  # type: ignore[attr-defined]
                analysis = await step._analyze_regime_characteristics(clustering, data_loaded["data"])  # type: ignore[attr-defined]
                result = {"stage": "analysis", "keys": list(analysis.keys())}

            elif only == "reports":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                hmm_res = await step._perform_hmm_regime_discovery_with_utilities(data_loaded["data"])  # type: ignore[attr-defined]
                clustering = await step._perform_final_clustering_with_utilities(data_loaded["data"], hmm_res)  # type: ignore[attr-defined]
                analysis = await step._analyze_regime_characteristics(clustering, data_loaded["data"])  # type: ignore[attr-defined]
                reports = await step._generate_comprehensive_reports(clustering, analysis)  # type: ignore[attr-defined]
                result = {"stage": "reports", "keys": list(reports.keys())}

            elif only == "save":
                data_loaded = await step._load_and_prepare_data()  # type: ignore[attr-defined]
                hmm_res = await step._perform_hmm_regime_discovery_with_utilities(data_loaded["data"])  # type: ignore[attr-defined]
                clustering = await step._perform_final_clustering_with_utilities(data_loaded["data"], hmm_res)  # type: ignore[attr-defined]
                analysis = await step._analyze_regime_characteristics(clustering, data_loaded["data"])  # type: ignore[attr-defined]
                reports = await step._generate_comprehensive_reports(clustering, analysis)  # type: ignore[attr-defined]
                saved = await step._save_final_results(clustering, analysis, reports)  # type: ignore[attr-defined]
                result = {"stage": "save", "saved": bool(saved)}

            else:
                raise ValueError(f"Unknown substep: {only}")

            # Export artifacts as in full run
            try:
                perf_summary = step.performance_monitor.get_summary() if hasattr(step, "performance_monitor") else {}
            except Exception:
                perf_summary = {}
            self._export_performance_summary(perf_summary)
            self._export_utility_status(step)
            self.export_function_call_report()

            # Persist lightweight substep result
            (self.output_dir / "substep_result.json").write_text(json.dumps(result, indent=2))

            elapsed = time.time() - start
            self.logger.info(f"✅ Substep '{only}' debug run completed in {elapsed:.2f}s | result_keys={list(result.keys())}")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Substep '{only}' debug run failed: {e}")
            try:
                self.export_function_call_report()
            except Exception:
                pass
            return False
        finally:
            try:
                await step.cleanup()
            except Exception:
                pass

    # ---------- Internals ----------
    def _export_performance_summary(self, performance_summary: Dict[str, Any]) -> Path:
        path = self.output_dir / "performance_summary.json"
        try:
            path.write_text(json.dumps(performance_summary or {}, indent=2))
            self.logger.info(f"📈 Performance summary exported: {path}")
        except Exception as e:
            self.logger.error(f"Failed to export performance summary: {e}")
        return path

    def _export_utility_status(self, step: FinalRegimeClusteringStep) -> Path:
        status: Dict[str, Any] = {}
        try:
            injector = getattr(step, "utility_injector", None)
            if injector and hasattr(injector, "get_initialization_status"):
                status = injector.get_initialization_status()  # type: ignore[assignment]
        except Exception:
            status = {}
        payload = {
            "timestamp": datetime.now().isoformat(),
            "injected_utilities_count": len(getattr(step, "utilities", {}) or {}),
            "initialization_status": status,
        }
        path = self.output_dir / "utility_injection_status.json"
        try:
            path.write_text(json.dumps(payload, indent=2))
            self.logger.info(f"🧩 Utility injection status exported: {path}")
        except Exception as e:
            self.logger.error(f"Failed to export utility status: {e}")
        return path


async def _run_async(debugger: Step03_5Debugger, config: Dict[str, Any], only: Optional[str]) -> bool:
    if only:
        return await debugger.run_substeps(config, only)
    return await debugger.run_full(config)


def run(config: Dict[str, Any], only: Optional[str] = None, verbose: bool = True, output_dir: Optional[str] = None) -> bool:
    """Synchronous entrypoint for debugging Step 3.5."""
    debugger = Step03_5Debugger(output_dir=output_dir)
    if verbose:
        debugger.enable_verbose_logging()
    debugger.export_environment_snapshot()

    try:
        # Prefer reusing an existing loop if present
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Run nested with ensuring task scheduling
            return asyncio.run(_run_async(debugger, config, only))
        else:
            return asyncio.run(_run_async(debugger, config, only))
    except Exception as e:
        # Final safety net
        logger = get_logger("Step03_5Debugger")
        logger.exception(f"Debug run failed: {e}")
        return False


__all__ = [
    "Step03_5Debugger",
    "run",
]

