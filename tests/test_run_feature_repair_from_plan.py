from scripts.run_feature_repair_from_plan import (
    MemorySnapshot,
    ProcessSnapshot,
    collect_process_snapshots,
    evaluate_safety,
    parse_process_table,
    parse_vm_stat,
    trusted_process_snapshots,
)


def test_parse_vm_stat_extracts_memory_values():
    snap = parse_vm_stat(
        """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages speculative:                         25.
Pages occupied by compressor:             50.
"""
    )

    assert snap.page_size == 16384
    assert round(snap.free_mb, 2) == 1.95
    assert round(snap.compressor_mb, 2) == 0.78


def test_parse_process_table_extracts_command_and_rss():
    procs = parse_process_table(
        "123 2048 python3 -m extreme_price_movements.inference.run_inference\n"
    )

    assert len(procs) == 1
    assert procs[0].pid == 123
    assert procs[0].rss_mb == 2.0
    assert "run_inference" in procs[0].command


def test_evaluate_safety_fails_closed_on_low_memory_and_hot_process():
    safe, reasons, payload = evaluate_safety(
        MemorySnapshot(
            page_size=16384,
            free_pages=10,
            speculative_pages=10,
            compressor_pages=500000,
        ),
        [
            ProcessSnapshot(
                pid=123,
                rss_mb=3000.0,
                command="python3 -m extreme_price_movements.inference.run_inference",
            )
        ],
        min_free_mb=1024.0,
        max_compressor_mb=4096.0,
        max_relevant_process_rss_mb=2048.0,
    )

    assert not safe
    assert any("free_memory_mb" in reason for reason in reasons)
    assert any("compressor_mb" in reason for reason in reasons)
    assert any("relevant_process_rss_mb" in reason for reason in reasons)
    assert payload["relevant_processes"][0]["pid"] == 123


def test_evaluate_safety_passes_when_limits_are_met():
    safe, reasons, _ = evaluate_safety(
        MemorySnapshot(
            page_size=16384,
            free_pages=100000,
            speculative_pages=10000,
            compressor_pages=10,
        ),
        [],
        min_free_mb=1024.0,
        max_compressor_mb=4096.0,
        max_relevant_process_rss_mb=2048.0,
    )

    assert safe
    assert reasons == []


def test_collect_process_snapshots_fails_closed_when_ps_is_unavailable(monkeypatch):
    def _raise(*args, **kwargs):
        raise PermissionError("blocked")

    monkeypatch.setattr("subprocess.check_output", _raise)

    procs = collect_process_snapshots()

    assert len(procs) == 1
    assert procs[0].pid == -1
    assert procs[0].rss_mb == 1_000_000_000.0
    assert "process_snapshot_unavailable" in procs[0].command


def test_trusted_process_snapshots_are_counted_as_relevant_processes():
    safe, reasons, payload = evaluate_safety(
        MemorySnapshot(
            page_size=16384,
            free_pages=100000,
            speculative_pages=10000,
            compressor_pages=10,
        ),
        trusted_process_snapshots(3000.0, "shell_ps"),
        min_free_mb=1024.0,
        max_compressor_mb=4096.0,
        max_relevant_process_rss_mb=2048.0,
    )

    assert not safe
    assert any("relevant_process_rss_mb" in reason for reason in reasons)
    assert payload["relevant_processes"][0]["rss_mb"] == 3000.0
    assert "shell_ps" in payload["relevant_processes"][0]["command"]
