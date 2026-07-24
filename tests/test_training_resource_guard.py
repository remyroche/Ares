from __future__ import annotations

import json
from pathlib import Path

import pytest

from extreme_price_movements.training_resource_guard import (
    ResourceSnapshot,
    ResourceTelemetryUnavailableError,
    TrainingResourceGuard,
    TrainingResourceLimitError,
    TrainingResourceLimits,
    _macos_available_ram_bytes,
)


class FakeClock:
    def __init__(self, value: float = 1_700_000_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _limits(**overrides: int | float | None) -> TrainingResourceLimits:
    values: dict[str, int | float | None] = {
        "min_free_ram_bytes": 100,
        "max_process_rss_bytes": 200,
        "min_free_disk_bytes": 300,
        "check_interval_seconds": 10.0,
    }
    values.update(overrides)
    return TrainingResourceLimits(**values)  # type: ignore[arg-type]


def test_preflight_accepts_sample_and_appends_jsonl_telemetry(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry" / "resources.jsonl"
    guard = TrainingResourceGuard(
        limits=_limits(),
        telemetry_path=telemetry_path,
        sampler=lambda _: ResourceSnapshot(
            available_ram_bytes=100, process_rss_bytes=200, free_disk_bytes=300
        ),
        clock=FakeClock(),
    )

    snapshot = guard.preflight("stage_a_context")

    assert snapshot == ResourceSnapshot(100, 200, 300)
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload == {
        "event": "preflight",
        "limits": {
            "check_interval_seconds": 10.0,
            "max_process_rss_bytes": 200,
            "min_free_disk_bytes": 300,
            "min_free_ram_bytes": 100,
        },
        "snapshot": {
            "available_ram_bytes": 100,
            "free_disk_bytes": 300,
            "process_rss_bytes": 200,
        },
        "stage": "stage_a_context",
        "status": "ok",
        "timestamp_utc": "2023-11-14T22:13:20+00:00",
    }


def test_preflight_fails_before_stage_when_any_limit_is_breached(
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "resources.jsonl"
    guard = TrainingResourceGuard(
        limits=_limits(),
        telemetry_path=telemetry_path,
        sampler=lambda _: ResourceSnapshot(
            available_ram_bytes=99, process_rss_bytes=201, free_disk_bytes=299
        ),
        clock=FakeClock(),
    )

    with pytest.raises(TrainingResourceLimitError) as raised:
        guard.preflight("stage_b_trigger_refinement")

    assert len(raised.value.violations) == 3
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert len(payload["violations"]) == 3


def test_missing_required_telemetry_fails_closed_and_is_recorded(
    tmp_path: Path,
) -> None:
    telemetry_path = tmp_path / "resources.jsonl"
    guard = TrainingResourceGuard(
        limits=_limits(),
        telemetry_path=telemetry_path,
        sampler=lambda _: ResourceSnapshot(
            available_ram_bytes=None, process_rss_bytes=1, free_disk_bytes=999
        ),
        clock=FakeClock(),
    )

    with pytest.raises(ResourceTelemetryUnavailableError, match="available_ram_bytes"):
        guard.preflight("stage_c_meta")

    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["missing_metrics"] == ["available_ram_bytes"]


def test_checkpoint_respects_interval_and_samples_again_when_due() -> None:
    clock = FakeClock(100.0)
    calls: list[Path] = []

    def sampler(disk_path: Path) -> ResourceSnapshot:
        calls.append(disk_path)
        return ResourceSnapshot(
            available_ram_bytes=100, process_rss_bytes=200, free_disk_bytes=300
        )

    guard = TrainingResourceGuard(limits=_limits(), sampler=sampler, clock=clock)
    assert guard.preflight("stage_a") is not None
    assert guard.checkpoint("stage_a") is None
    clock.value = 109.9
    assert guard.checkpoint("stage_a") is None
    clock.value = 110.0
    assert guard.checkpoint("stage_a") == ResourceSnapshot(100, 200, 300)
    assert calls == [Path("."), Path(".")]


def test_disabled_limit_does_not_require_its_metric() -> None:
    guard = TrainingResourceGuard(
        limits=_limits(min_free_ram_bytes=None),
        sampler=lambda _: ResourceSnapshot(
            available_ram_bytes=None, process_rss_bytes=200, free_disk_bytes=300
        ),
        clock=FakeClock(),
    )

    assert guard.preflight("stage_a") == ResourceSnapshot(None, 200, 300)


def test_macos_vm_stat_fallback_uses_reported_page_size(monkeypatch) -> None:
    output = "\n".join(
        [
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
            "Pages free: 10.",
            "Pages inactive: 20.",
            "Pages speculative: 5.",
        ]
    )
    monkeypatch.setattr(
        "extreme_price_movements.training_resource_guard.subprocess.check_output",
        lambda *args, **kwargs: output,
    )

    assert _macos_available_ram_bytes() == 35 * 16384


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_free_ram_bytes": -1},
        {"max_process_rss_bytes": 1.5},
        {"min_free_disk_bytes": -1},
        {"check_interval_seconds": -0.1},
    ],
)
def test_limits_reject_invalid_configuration(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        TrainingResourceLimits(**kwargs)
