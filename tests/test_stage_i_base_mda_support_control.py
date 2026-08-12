from __future__ import annotations

import numpy as np

from extreme_price_movements.stage_i_mda_support import (
    restrict_stage_i_mda_training_support,
)


def test_target_only_mda_support_removes_realised_path_context() -> None:
    source = {
        "label_context": {
            "side_name": np.asarray(["long", "long"], dtype=object),
            "valid_resolved_support": np.asarray([1.0, 1.0], dtype=np.float32),
            "feature_selection_archetype": np.asarray(["fast", "slow"], dtype=object),
            "r3_class": np.asarray([2, 0], dtype=np.int8),
            "event_upper": np.asarray([1.0, 0.0], dtype=np.float32),
            "exact_net_bps": np.asarray([125.0, -200.0], dtype=np.float32),
            "path_economic_state": np.asarray(["clear", "adverse"], dtype=object),
        },
        "archetype_labels": np.asarray(["fast", "slow"], dtype=object),
        "audit": {"rows": 2},
    }

    control, audit = restrict_stage_i_mda_training_support(source, mode="target-only")

    assert set(control) == {"side_name", "valid_resolved_support"}
    assert audit["mode"] == "target-only"
    assert audit["realised_path_support_available"] is False
    assert "feature_selection_archetype" in audit["removed_label_context_fields"]
    assert "exact_net_bps" in audit["removed_label_context_fields"]


def test_full_mda_support_preserves_original_context() -> None:
    context = {
        "side_name": np.asarray(["short"], dtype=object),
        "valid_resolved_support": np.asarray([1.0], dtype=np.float32),
        "feature_selection_archetype": np.asarray(["timeout"], dtype=object),
    }
    source = {"label_context": context, "archetype_labels": None, "audit": {}}

    actual, audit = restrict_stage_i_mda_training_support(source, mode="full")

    assert actual is context
    assert audit["realised_path_support_available"] is True
    assert audit["archetype_conditioned_enabled"] is True
