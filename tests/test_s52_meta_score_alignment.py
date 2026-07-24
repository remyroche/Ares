import numpy as np

from extreme_price_movements.inference.s52_meta_score_alignment import (
    apply_s52_meta_score_alignment,
    fit_paired_s52_meta_score_alignment,
)
from scripts.refit_package_s52_meta_shared_champion import _resolve_final_fit_shards


def test_paired_alignment_is_monotonic_and_reduces_same_row_error() -> None:
    source = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    target = np.clip(0.1 + 0.75 * source + 0.05 * source**2, 0.0, 1.0)
    alignment = fit_paired_s52_meta_score_alignment(
        {"long": source}, {"long": target}, minimum_rows=64
    )
    mapped = apply_s52_meta_score_alignment(source, alignment, side="long")
    assert alignment["mode"] == "side_specific_same_row_isotonic_bridge"
    assert np.all(np.diff(mapped) >= 0.0)
    assert np.mean(np.abs(mapped - target)) < np.mean(np.abs(source - target))


def test_paired_alignment_keeps_sides_independent() -> None:
    source = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    alignment = fit_paired_s52_meta_score_alignment(
        {"long": source, "short": source},
        {"long": source * 0.5, "short": 0.5 + source * 0.5},
        minimum_rows=64,
    )
    probe = np.array([0.5], dtype=np.float32)
    long_value = apply_s52_meta_score_alignment(probe, alignment, side="long")[0]
    short_value = apply_s52_meta_score_alignment(probe, alignment, side="short")[0]
    assert long_value < short_value


def test_full_final_fit_uses_one_complete_dataset() -> None:
    chronological = [(0, 10), (10, 20), (20, 30)]
    assert _resolve_final_fit_shards(
        mode="full_dataset", row_count=30, chronological_shards=chronological
    ) == [(0, 30)]
    assert _resolve_final_fit_shards(
        mode="chronological_shards_legacy",
        row_count=30,
        chronological_shards=chronological,
    ) == chronological
