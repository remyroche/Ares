"""Fail-closed resource checks for memory- and disk-intensive training stages.

``TrainingResourceGuard`` is intentionally synchronous: a training runner calls
``preflight`` before allocating a large workload and ``checkpoint`` at natural
stage boundaries.  There is no background thread, so a guard cannot keep a
process alive or hide failures from the caller.

The defaults reserve 2 GiB of immediately available RAM, cap this process at
12 GiB RSS, and reserve 10 GiB of free disk.  They are conservative starting
limits for roadmap-heavy local training, not an estimate of a model's needs;
stage owners should set explicit limits when their artifact size is known.
Unavailable required telemetry is an error rather than a pass.  This preserves
the fail-closed contract on platforms where optional telemetry support is not
installed or is restricted.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

GIB = 1024**3


@dataclass(frozen=True)
class ResourceSnapshot:
    """One sample of resources required by a training stage.

    A value of ``None`` means the collector could not obtain that metric.  The
    guard rejects a missing value whenever its corresponding limit is enabled.
    """

    available_ram_bytes: int | None
    process_rss_bytes: int | None
    free_disk_bytes: int | None


@dataclass(frozen=True)
class TrainingResourceLimits:
    """Limits and cadence for :class:`TrainingResourceGuard`.

    ``None`` disables one specific limit.  Enabled limits use bytes and have
    safe, conservative defaults: 2 GiB free RAM, 12 GiB process RSS, and
    10 GiB free disk.  Checkpoints sample at most once per interval unless a
    caller uses ``preflight``, which always samples immediately.
    """

    min_free_ram_bytes: int | None = 2 * GIB
    max_process_rss_bytes: int | None = 12 * GIB
    min_free_disk_bytes: int | None = 10 * GIB
    check_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in (
            "min_free_ram_bytes",
            "max_process_rss_bytes",
            "min_free_disk_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if self.check_interval_seconds < 0:
            raise ValueError("check_interval_seconds must be non-negative")


class TrainingResourceGuardError(RuntimeError):
    """Base exception for a guard that cannot safely approve training."""


class ResourceTelemetryUnavailableError(TrainingResourceGuardError):
    """Raised when an enabled resource limit cannot be measured."""

    def __init__(self, metric: str) -> None:
        self.metric = metric
        super().__init__(
            f"Required training resource telemetry is unavailable: {metric}"
        )


class TrainingResourceLimitError(TrainingResourceGuardError):
    """Raised before work continues when the current sample violates a limit."""

    def __init__(self, violations: tuple[str, ...], snapshot: ResourceSnapshot) -> None:
        self.violations = violations
        self.snapshot = snapshot
        super().__init__("Training resource limits exceeded: " + "; ".join(violations))


class TrainingResourceTelemetryWriteError(TrainingResourceGuardError):
    """Raised when configured audit telemetry cannot be durably appended."""


ResourceSampler = Callable[[Path], ResourceSnapshot]
Clock = Callable[[], float]


def default_resource_sampler(disk_path: Path) -> ResourceSnapshot:
    """Collect available RAM, current-process RSS, and free disk cross-platform.

    ``psutil`` is preferred because it exposes all three current measurements
    on supported operating systems.  Native fallbacks cover common Linux,
    macOS, and Windows environments.  Any unsupported measurement is returned
    as ``None`` so an enabled guard limit rejects it instead of silently
    continuing.
    """

    try:
        import psutil

        return ResourceSnapshot(
            available_ram_bytes=int(psutil.virtual_memory().available),
            process_rss_bytes=int(psutil.Process(os.getpid()).memory_info().rss),
            free_disk_bytes=int(shutil.disk_usage(disk_path).free),
        )
    except Exception:
        return ResourceSnapshot(
            available_ram_bytes=_available_ram_bytes_fallback(),
            process_rss_bytes=_process_rss_bytes_fallback(),
            free_disk_bytes=_free_disk_bytes(disk_path),
        )


def _free_disk_bytes(disk_path: Path) -> int | None:
    try:
        return int(shutil.disk_usage(disk_path).free)
    except OSError:
        return None


def _available_ram_bytes_fallback() -> int | None:
    if os.name == "nt":
        return _windows_memory_status("ullAvailPhys")

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages >= 0:
            return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        pass

    if platform.system() == "Darwin":
        return _macos_available_ram_bytes()
    return None


def _macos_available_ram_bytes() -> int | None:
    """Best-effort macOS fallback when psutil and ``SC_AVPHYS_PAGES`` fail."""

    try:
        output = subprocess.check_output(
            ["vm_stat"], text=True, stderr=subprocess.DEVNULL
        )
        first_line = output.splitlines()[0] if output else ""
        page_size_match = re.search(r"page size of (\d+) bytes", first_line)
        if page_size_match is None:
            return None
        page_size = int(page_size_match.group(1))
        if page_size <= 0:
            return None
        pages: dict[str, int] = {}
        for line in output.splitlines()[1:]:
            key, _, raw_value = line.partition(":")
            value = raw_value.strip().rstrip(".")
            if value.isdigit():
                pages[key.strip()] = int(value)
        # Free, inactive, and speculative pages are readily reclaimable.
        return page_size * sum(
            pages.get(key, 0)
            for key in ("Pages free", "Pages inactive", "Pages speculative")
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _process_rss_bytes_fallback() -> int | None:
    if os.name == "nt":
        return _windows_process_rss_bytes()

    try:
        statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        resident_pages = int(statm[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError, AttributeError):
        pass

    # macOS exposes only peak resident size through resource; it is a safe
    # upper bound when current RSS is unavailable, so it cannot under-report.
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak < 0:
            return None
        return peak if platform.system() == "Darwin" else peak * 1024
    except (ImportError, OSError, ValueError):
        return None


def _windows_memory_status(field: str) -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(getattr(status, field))
    except (AttributeError, OSError):
        return None


def _windows_process_rss_bytes() -> int | None:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.WorkingSetSize)
    except (AttributeError, OSError):
        return None


class TrainingResourceGuard:
    """Synchronous fail-closed preflight and checkpoint monitor for training.

    The guard is deterministic under injected ``sampler`` and ``clock``.  The
    clock should return non-decreasing Unix seconds; a backwards step forces a
    fresh checkpoint sample rather than extending the check interval.
    """

    def __init__(
        self,
        *,
        limits: TrainingResourceLimits | None = None,
        disk_path: str | Path = ".",
        telemetry_path: str | Path | None = None,
        sampler: ResourceSampler = default_resource_sampler,
        clock: Clock = time.time,
    ) -> None:
        self.limits = limits or TrainingResourceLimits()
        self.disk_path = Path(disk_path)
        self.telemetry_path = (
            Path(telemetry_path) if telemetry_path is not None else None
        )
        self._sampler = sampler
        self._clock = clock
        self._last_check_at: float | None = None

    def preflight(self, stage: str) -> ResourceSnapshot:
        """Sample and validate resources immediately before beginning ``stage``."""

        return self._check(stage=stage, event="preflight")

    def checkpoint(self, stage: str) -> ResourceSnapshot | None:
        """Validate a stage boundary when the configured interval has elapsed.

        Returns ``None`` when the last sample is still fresh.  This lets callers
        place checkpoints inside loops without creating a polling daemon.
        """

        now = self._clock()
        if self._last_check_at is not None:
            elapsed = now - self._last_check_at
            if 0 <= elapsed < self.limits.check_interval_seconds:
                return None
        return self._check(stage=stage, event="checkpoint", now=now)

    def _check(
        self,
        *,
        stage: str,
        event: Literal["preflight", "checkpoint"],
        now: float | None = None,
    ) -> ResourceSnapshot:
        if not stage or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        checked_at = self._clock() if now is None else now
        try:
            snapshot = self._sampler(self.disk_path)
        except Exception as exc:
            self._write_event(
                stage=stage,
                event=event,
                checked_at=checked_at,
                snapshot=None,
                status="telemetry_unavailable",
                detail={"reason": f"sampler failed: {type(exc).__name__}"},
            )
            raise ResourceTelemetryUnavailableError("resource sampler") from exc

        missing = self._missing_required_metrics(snapshot)
        violations = self._limit_violations(snapshot) if not missing else ()
        status = "ok" if not missing and not violations else "rejected"
        detail: dict[str, object] = {}
        if missing:
            detail["missing_metrics"] = missing
        if violations:
            detail["violations"] = violations
        self._write_event(
            stage=stage,
            event=event,
            checked_at=checked_at,
            snapshot=snapshot,
            status=status,
            detail=detail,
        )
        self._last_check_at = checked_at
        if missing:
            raise ResourceTelemetryUnavailableError(", ".join(missing))
        if violations:
            raise TrainingResourceLimitError(violations, snapshot)
        return snapshot

    def _missing_required_metrics(self, snapshot: ResourceSnapshot) -> tuple[str, ...]:
        required = (
            ("available_ram_bytes", self.limits.min_free_ram_bytes),
            ("process_rss_bytes", self.limits.max_process_rss_bytes),
            ("free_disk_bytes", self.limits.min_free_disk_bytes),
        )
        return tuple(
            name
            for name, limit in required
            if limit is not None and getattr(snapshot, name) is None
        )

    def _limit_violations(self, snapshot: ResourceSnapshot) -> tuple[str, ...]:
        violations: list[str] = []
        if (
            self.limits.min_free_ram_bytes is not None
            and snapshot.available_ram_bytes is not None
            and snapshot.available_ram_bytes < self.limits.min_free_ram_bytes
        ):
            violations.append(
                "available_ram_bytes="
                f"{snapshot.available_ram_bytes} is below min_free_ram_bytes="
                f"{self.limits.min_free_ram_bytes}"
            )
        if (
            self.limits.max_process_rss_bytes is not None
            and snapshot.process_rss_bytes is not None
            and snapshot.process_rss_bytes > self.limits.max_process_rss_bytes
        ):
            violations.append(
                "process_rss_bytes="
                f"{snapshot.process_rss_bytes} exceeds max_process_rss_bytes="
                f"{self.limits.max_process_rss_bytes}"
            )
        if (
            self.limits.min_free_disk_bytes is not None
            and snapshot.free_disk_bytes is not None
            and snapshot.free_disk_bytes < self.limits.min_free_disk_bytes
        ):
            violations.append(
                "free_disk_bytes="
                f"{snapshot.free_disk_bytes} is below min_free_disk_bytes="
                f"{self.limits.min_free_disk_bytes}"
            )
        return tuple(violations)

    def _write_event(
        self,
        *,
        stage: str,
        event: str,
        checked_at: float,
        snapshot: ResourceSnapshot | None,
        status: str,
        detail: dict[str, object],
    ) -> None:
        if self.telemetry_path is None:
            return
        payload: dict[str, object] = {
            "timestamp_utc": datetime.fromtimestamp(checked_at, UTC).isoformat(),
            "event": event,
            "stage": stage,
            "status": status,
            "limits": asdict(self.limits),
            "snapshot": asdict(snapshot) if snapshot is not None else None,
            **detail,
        }
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
                handle.flush()
        except OSError as exc:
            raise TrainingResourceTelemetryWriteError(
                f"Unable to append training resource telemetry to {self.telemetry_path}"
            ) from exc
