from __future__ import annotations

from types import SimpleNamespace

from extreme_price_movements.inference.run_inference import (
    _canonical_meta_postprocessor_for_side,
)


def test_canonical_postprocessor_can_be_restricted_to_short_route() -> None:
    postprocessor = object()
    policy = SimpleNamespace(canonical_meta_postprocessor_sides=("short",))

    assert (
        _canonical_meta_postprocessor_for_side(postprocessor, policy, "short")
        is postprocessor
    )
    assert _canonical_meta_postprocessor_for_side(postprocessor, policy, "long") is None


def test_empty_postprocessor_side_contract_preserves_shared_behavior() -> None:
    postprocessor = object()
    policy = SimpleNamespace(canonical_meta_postprocessor_sides=())

    assert (
        _canonical_meta_postprocessor_for_side(postprocessor, policy, "long")
        is postprocessor
    )
    assert (
        _canonical_meta_postprocessor_for_side(postprocessor, policy, "short")
        is postprocessor
    )
