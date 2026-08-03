from pathlib import Path

import pytest

from scripts.seal_stage_c_stage0_artifact import compute_output_hashes, verify_output_hashes


def test_output_hash_verification_fails_closed_on_mismatch(tmp_path: Path) -> None:
    (tmp_path / "feature_panel.parquet").write_bytes(b"frozen-feature-bytes")
    (tmp_path / "correctness_test_report.json").write_bytes(b'{"passed": true}\n')
    (tmp_path / "run_manifest.json").write_bytes(b"manifest is deliberately excluded")

    expected = compute_output_hashes(tmp_path)
    assert set(expected) == {"correctness_test_report.json", "feature_panel.parquet"}
    assert verify_output_hashes(tmp_path, expected) == {
        "verified": True,
        "output_count": 2,
        "correctness_report_included": True,
        "manifest_excluded": True,
    }

    (tmp_path / "feature_panel.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"mismatched=\['feature_panel.parquet'\]"):
        verify_output_hashes(tmp_path, expected)


def test_output_hash_verification_fails_closed_on_unexpected_file(tmp_path: Path) -> None:
    (tmp_path / "correctness_test_report.json").write_bytes(b"{}\n")
    expected = compute_output_hashes(tmp_path)
    (tmp_path / "unsealed.txt").write_bytes(b"not declared")

    with pytest.raises(ValueError, match=r"unexpected=\['unsealed.txt'\]"):
        verify_output_hashes(tmp_path, expected)
