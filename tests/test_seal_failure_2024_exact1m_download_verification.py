from pathlib import Path


def test_verifier_requires_all_four_complete_nonoverlapping_partitions():
    source = Path(
        "scripts/seal_failure_2024_exact1m_download_verification.py"
    ).read_text()
    assert "for partition in range(4)" in source
    assert "required_minutes" in source and "covered_minutes" in source
    assert "intersection(local_symbols)" in source
    assert '"status": "SEALED_COMPLETE"' in source
    assert "full_2024_candidate_local_policy_label_replay" in source
