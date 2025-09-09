"""
Step03 Debug Tools

Comprehensive debugging utilities for Step 3 (HMM regime discovery).

Provides:
- Import/dependency verification
- Data presence and schema checks
- Artifact validation (leveraging existing validator)
- Environment resource snapshot (CPU/memory/disk)
- Optional smoke test of the enhanced Step03 runner with timeout
- JSON report aggregation and saving convenience
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import system_logger
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.pipeline_standards import pipeline_standards

# Optional imports guarded at runtime
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def _now_iso() -> str:
    return datetime.now().isoformat(timespec = 'seconds')


def _safe_import(module: str) -> tuple[bool, str]:
    try:
        __import__(module)
        return True, ''
    except Exception as e:  # pragma: no cover - environment dependent
        return False, str(e)


def _get_pkg_version(pkg_name: str) -> Optional[str]:
    try:
        mod = __import__(pkg_name)
        return getattr(mod, '__version__', None)
    except Exception:  # pragma: no cover
        return None


@dataclass
class ImportCheckResult:
    core_package: str
    ok: bool
    error: str | None = None


@dataclass
class DependencySummary:
    packages: Dict[str, Dict[str, Any]] = field(default_factory = dict)


@dataclass
class DataCheckResult:
    base_dir: str
    required: List[str]
    missing: List[str]
    present: Dict[str, Dict[str, Any]]


@dataclass
class ArtifactCheckResult:
    required: List[str]
    missing: List[str]
    info: Dict[str, Dict[str, Any]]
    validation_passed: bool


@dataclass
class ResourceSnapshot:
    cpu_percent: Optional[float]
    memory_percent: Optional[float]
    rss_mb: Optional[float]
    disk_percent_root: Optional[float]


@dataclass
class SmokeTestResult:
    attempted: bool
    completed: bool
    within_timeout: bool
    success_signal: Optional[bool]
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


@dataclass
class Step03DebugReport:
    timestamp: str
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    dependency_summary: DependencySummary
    import_checks: List[ImportCheckResult]
    data_check: DataCheckResult
    artifact_check: ArtifactCheckResult
    resource_snapshot: ResourceSnapshot
    smoke_test: SmokeTestResult
    suggestions: List[str] = field(default_factory = list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_import_checks() -> List[ImportCheckResult]:
    logger = system_logger.getChild('Step03DebugTools.ImportChecks')
    checks: List[tuple[str, str]] = [
        ('src.core.decorators', 'core decorators package'),
        ('src.core.decorators.function_monitor', 'function monitor'),
        ('src.core.decorators.enhanced_error_handling', 'enhanced error handling'),
        ('src.training.steps.market_analysis.step03_hmm_clustering', 'step03 main'),
        ('src.training.steps.market_analysis.hmm_clustering.step03_enhanced_hmm_regime_discovery', 'enhanced step03 runner'),
        ('src.training.steps.market_analysis.hmm_clustering.step03_hmm_regime_discovery_validator', 'step03 validator'),
    ]
    results: List[ImportCheckResult] = []
    for module, _desc in checks:
        ok, err = _safe_import(module)
        results.append(ImportCheckResult(core_package = module, ok = ok, error = (err if not ok else None)))
        if not ok:
            logger.error(f'Import failed: {module} -> {err}')
    return results


def summarize_dependencies() -> DependencySummary:
    packages = {
        'python': {'version': sys.version},
        'pandas': {'version': _get_pkg_version('pandas')},
        'numpy': {'version': _get_pkg_version('numpy')},
        'psutil': {'version': _get_pkg_version('psutil')},
        'hmmlearn': {'version': _get_pkg_version('hmmlearn')},
        'lightgbm': {'version': _get_pkg_version('lightgbm')},
        'optuna': {'version': _get_pkg_version('optuna')},
        'sklearn': {'version': _get_pkg_version('sklearn')},
    }
    return DependencySummary(packages = packages)


def check_data_presence(symbol: str, exchange: str, timeframe: str, data_dir: Optional[str]) -> DataCheckResult:
    logger = system_logger.getChild('Step03DebugTools.DataChecks')
    # Prefer processed_data directory used by step03
    base_dir = Path(data_dir) if data_dir else Path(pipeline_standards.build_path('processed_data', exchange, symbol))
    base_dir.mkdir(parents = True, exist_ok = True)

    # Multiple candidate patterns to increase robustness across repos
    candidates = [
        f"{exchange}_{symbol}_processed.parquet",
        f"{exchange}_{symbol}_volume_consolidated.parquet",
        pipeline_standards.generate_file_name('klines', exchange, symbol, timeframe),
        pipeline_standards.generate_file_name('validated_data', exchange, symbol, timeframe),
    ]

    present: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for fname in candidates:
        fpath = base_dir / fname
        if fpath.exists():
            try:
                stat = fpath.stat()
                present[fname] = {
                    'path': str(fpath),
                    'size_bytes': stat.st_size,
                    'modified_time': stat.st_mtime,
                }
            except Exception as e:  # pragma: no cover
                logger.warning(f'Could not stat file {fpath}: {e}')
                present[fname] = {'path': str(fpath)}
        else:
            missing.append(fname)

    return DataCheckResult(base_dir = str(base_dir), required = candidates, missing = missing, present = present)


async def validate_artifacts(symbol: str, exchange: str, timeframe: str, data_dir: str) -> ArtifactCheckResult:
    try:
        from .step03_hmm_regime_discovery_validator import run_validator  # type: ignore
    except Exception as e:
        return ArtifactCheckResult(required = [], missing = [f'validator import failed: {e}'], info = {}, validation_passed = False)

    training_input = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
    pipeline_state = {'hmm_regime_discovery': {'completed': True}}
    result = await run_validator(training_input, pipeline_state)  # type: ignore

    required = result.get('total_artifacts', 0)
    missing_list = result.get('missing_artifacts', [])
    artifact_info = result.get('artifact_info', {})
    return ArtifactCheckResult(
        required = ['count=' + str(required)],
        missing = list(missing_list) if isinstance(missing_list, list) else [],
        info = dict(artifact_info) if isinstance(artifact_info, dict) else {},
        validation_passed = bool(result.get('validation_passed', False)),
    )


def snapshot_resources() -> ResourceSnapshot:
    if psutil is None:  # pragma: no cover
        return ResourceSnapshot(cpu_percent = None, memory_percent = None, rss_mb = None, disk_percent_root = None)
    try:
        process = psutil.Process()
        mem = process.memory_info().rss / 1024 / 1024
        return ResourceSnapshot(
            cpu_percent = float(psutil.cpu_percent(interval = 0.2)),
            memory_percent = float(psutil.virtual_memory().percent),
            rss_mb = float(mem),
            disk_percent_root = float(psutil.disk_usage('/').percent),
        )
    except Exception:  # pragma: no cover
        return ResourceSnapshot(cpu_percent = None, memory_percent = None, rss_mb = None, disk_percent_root = None)


async def run_smoke_test(symbol: str, exchange: str, timeframe: str, data_dir: Optional[str], timeout_seconds: float = 30.0, enabled: bool = True) -> SmokeTestResult:
    if not enabled:
        return SmokeTestResult(attempted = False, completed = False, within_timeout = True, success_signal = None)

    try:
        from .step03_enhanced_hmm_regime_discovery import run_enhanced_step  # type: ignore
    except Exception as e:
        return SmokeTestResult(attempted = True, completed = False, within_timeout = True, success_signal = None, error = f'Import failed: {e}')

    start = time.time()
    async def _go() -> bool:
        return await run_enhanced_step(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = (data_dir or None), force_rerun = False, n_trials = 2, timeout_minutes = 1, cv_folds = 2)  # type: ignore

    try:
        ok = await asyncio.wait_for(_go(), timeout = timeout_seconds)
        return SmokeTestResult(attempted = True, completed = True, within_timeout = True, success_signal = bool(ok), duration_seconds = time.time() - start)
    except asyncio.TimeoutError:
        return SmokeTestResult(attempted = True, completed = False, within_timeout = False, success_signal = None, error = f'timed out after {timeout_seconds:.1f}s', duration_seconds = time.time() - start)
    except Exception as e:
        return SmokeTestResult(attempted = True, completed = False, within_timeout = True, success_signal = None, error = str(e), duration_seconds = time.time() - start)


async def run_debug_suite(symbol: str = 'ETHUSDT', exchange: str = 'BINANCE', timeframe: str = '1m', data_dir: Optional[str] = None, *, smoke_test: bool = False, smoke_timeout_seconds: float = 30.0) -> Step03DebugReport:
    logger = system_logger.getChild('Step03DebugTools')
    logger.info('🔧 Running Step03 debug suite...')

    imports = run_import_checks()
    deps = summarize_dependencies()

    data_check = check_data_presence(symbol, exchange, timeframe, data_dir)
    artifacts = await validate_artifacts(symbol, exchange, timeframe, data_check.base_dir)
    resources = snapshot_resources()
    smoke = await run_smoke_test(symbol, exchange, timeframe, data_check.base_dir, timeout_seconds = smoke_timeout_seconds, enabled = smoke_test)

    suggestions: List[str] = []
    # Suggestions based on findings
    missing_any = bool(data_check.missing)
    if missing_any:
        suggestions.append('Ensure preprocessed files exist in processed_data directory or provide --data-dir.')
        suggestions.append('Consider running earlier steps to generate processed inputs.')
    if not artifacts.validation_passed:
        suggestions.append('Artifacts validation failed; check Step03 execution and validator expectations.')
    for ic in imports:
        if not ic.ok:
            suggestions.append(f'Import failed: {ic.core_package} -> {ic.error}')
    if smoke.attempted and (not smoke.completed or not smoke.success_signal):
        suggestions.append('Smoke test failed or timed out; inspect logs for details.')

    report = Step03DebugReport(
        timestamp = _now_iso(),
        symbol = symbol,
        exchange = exchange,
        timeframe = timeframe,
        data_dir = data_check.base_dir,
        dependency_summary = deps,
        import_checks = imports,
        data_check = data_check,
        artifact_check = artifacts,
        resource_snapshot = resources,
        smoke_test = smoke,
        suggestions = suggestions,
    )
    logger.info('✅ Step03 debug suite completed')
    return report


def save_report(report: Step03DebugReport, output_path: str | os.PathLike) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w') as f:
        json.dump(report.to_dict(), f, indent = 2)
    return str(path)


# Convenience entry for programmatic use
async def debug_and_save(symbol: str = 'ETHUSDT', exchange: str = 'BINANCE', timeframe: str = '1m', data_dir: Optional[str] = None, *, smoke_test: bool = False, output_dir: str = 'results') -> str:
    report = await run_debug_suite(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, smoke_test = smoke_test)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = Path(output_dir) / f'step03_debug_report_{symbol}_{timeframe}_{ts}.json'
    return save_report(report, out_file)

