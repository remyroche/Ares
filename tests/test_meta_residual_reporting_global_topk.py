from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_train_meta_residual_archetype_enhancement import metrics_by_scope


def test_metrics_decompose_one_global_timestamp_topk_selection() -> None:
    timestamps = pd.to_datetime(
        ["2026-04-01T00:00:00Z"] * 4 + ["2026-04-01T01:00:00Z"] * 4,
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": [f"S{idx}" for idx in range(8)],
            "calendar_month": "2026-04",
            "week_start": pd.Timestamp("2026-03-30", tz="UTC"),
            "side_name": ["long", "short", "short", "short"] * 2,
            "archetype_policy_key": ["long_a", "short_a", "short_b", "short_c"] * 2,
            # Alternative selects the one long globally at each timestamp.
            "score_alternative": [0.99, 0.80, 0.70, 0.60] * 2,
            # Reference selects short_a globally at each timestamp.
            "score_current_reference": [0.60, 0.99, 0.80, 0.70] * 2,
            "hit_prob_alternative": np.full(8, 0.5, dtype=np.float32),
            "hit_prob_current_reference": np.full(8, 0.5, dtype=np.float32),
            "ev_after_1pct": np.linspace(-0.01, 0.02, 8),
            "clean_exec": [1, 0, 0, 0] * 2,
            "dirty_positive": [0, 1, 1, 1] * 2,
            "first_touch_bad_mae_1r": [0, 1, 1, 1] * 2,
            "full_path_bad_mae_1r": [0, 1, 1, 1] * 2,
            "timeout": np.zeros(8, dtype=np.float32),
        }
    )
    metrics = metrics_by_scope(frame, "local_aegmm_all_three")
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq("local_aegmm_all_three")
    ].iloc[0]
    assert int(overall["selected_rows"]) == 2
    assert overall["selection_basis"] == "global_within_timestamp"
    side = metrics[
        metrics["scope"].eq("side")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq("local_aegmm_all_three")
    ].set_index("side_name")
    assert int(side.loc["long", "selected_rows"]) == 2
    assert int(side.loc["short", "selected_rows"]) == 0
