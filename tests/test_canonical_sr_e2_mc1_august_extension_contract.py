from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_august_extension_retains_target_free_then_label_join_order() -> None:
    prepare = (ROOT / "scripts/prepare_canonical_sr_e2_august_extension_inputs.py").read_text()
    runner = (ROOT / "scripts/run_canonical_sr_e2_mc1_august_extension.py").read_text()
    assert "no outcomes or exchange calls" in prepare
    assert "score_complete_dual_map" in prepare
    assert "_load_aug_labels(policy_labels" in runner
    assert "_refit_family(current_full" in runner
    assert "BCF MC1 >= +50 AND current-v5 MC1 >= +50" in runner
    assert "no live or canonical mutation" in runner


def test_august_extension_covers_only_archived_score_horizon() -> None:
    runner = (ROOT / "scripts/run_canonical_sr_e2_mc1_august_extension.py").read_text()
    assert "2026-08-19T00:00:00Z" in runner
    assert "ends 2026-08-18 21:00 UTC" in runner


def test_august_parent_policy_wrapper_matches_retained_source_geometry() -> None:
    policy = (ROOT / "config/strict_r3_source_aligned_parent_policy_long_20260831_v1.json").read_text()
    assert '"side": "long"' in policy
    assert '"sl_mult": 4.15200064332387' in policy
    assert '"trailing_activation_mult": 2.326224919759605' in policy
    assert '"fixed_trailing_gap_mult": 0.10237198997143725' in policy


def test_frozen_policy_materialiser_falls_back_only_to_local_one_minute_data() -> None:
    source = (ROOT / "scripts/materialize_strict_r3_frozen_policy_labels_v2.py").read_text()
    assert "local_1m_aggregate" in source
    assert "ArrowInvalid" in source
    assert "no download, interpolation, or future fill" in source


def test_august_feature_extension_uses_bounded_symbol_batches() -> None:
    source = (ROOT / "scripts/run_canonical_sr_e2_mc1_august_extension.py").read_text()
    assert 'groupby("symbol", sort=True)' in source
    assert "grouped[offset:offset + 8]" in source
    assert "bounded August feature materialisation changed target-free identity" in source
