from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_geometry_path_quality_cost_ladder import (  # noqa: E402
    adjust_geometry_cost,
    build_cost_ladder,
)


def _row(
    arm: str,
    *,
    top_frac: float = 0.10,
    fit_evw: float = 0.72,
    gross: float = 0.006,
    first_touch_bad: float = 0.15,
    mae_before: float = 0.20,
    underwater: float = 8.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "sweep",
        "arm": arm,
        "selection_mode": "global",
        "top_frac": top_frac,
        "regime_family": "all",
        "tp_r": 1.0,
        "sl_r": 0.5,
        "trail_r": 0.5,
        "max_bars_to_mfe": 12.0,
        "max_barrier": 0.05,
    }
    for period in ("all", "fit", "holdout"):
        row.update(
            {
                f"{period}_selected_rows": 1000,
                f"{period}_min_side_selected_rows": 200,
                f"{period}_gross_ev_weighted_first_touch_precision": fit_evw,
                f"{period}_min_side_gross_ev_weighted_first_touch_precision": fit_evw - 0.05,
                f"{period}_mean_capture_gross": gross,
                f"{period}_mean_capture_net": gross - 0.01,
                f"{period}_first_touch_bad_mae_1r_rate": first_touch_bad,
                f"{period}_selected_path_bad_mae_1r_rate": 0.55,
                f"{period}_first_touch_p90_mae_norm": 1.5,
                f"{period}_selected_path_p90_mae_norm": 8.0,
                f"{period}_mae_1r_before_mfe_1r_rate": mae_before,
                f"{period}_mfe_1r_before_mae_1r_rate": 1.0 - mae_before,
                f"{period}_mean_max_adverse_before_mfe_1r": 1.2,
                f"{period}_mean_underwater_bars_before_mfe_1r": underwater,
                f"{period}_mean_underwater_fraction_before_mfe_1r": 0.30,
                f"{period}_timeout_rate": 0.03,
            }
        )
    return row


def test_adjust_geometry_cost_recomputes_net_from_gross() -> None:
    frame = pd.DataFrame([_row("a", gross=0.006)])

    adjusted = adjust_geometry_cost(frame, cost=0.0025)

    assert adjusted.iloc[0]["fit_mean_capture_net"] == 0.0035
    assert adjusted.iloc[0]["holdout_mean_capture_net"] == 0.0035
    assert adjusted.iloc[0]["cost_bps"] == 25.0


def test_cost_ladder_keeps_separate_cost_specific_selections() -> None:
    candidates = pd.DataFrame(
        [
            _row("lower_gross_clean", fit_evw=0.72, gross=0.0025, first_touch_bad=0.12),
            _row("higher_gross_slightly_dirtier", fit_evw=0.70, gross=0.0090, first_touch_bad=0.20),
        ]
    )

    expanded, selected = build_cost_ladder(
        candidates,
        costs=[0.0, 0.01],
        top_fracs=[0.10],
        min_fit_rows=100,
        min_fit_side_rows=50,
    )

    assert sorted(expanded["cost_bps"].unique().tolist()) == [0.0, 100.0]
    assert selected["cost_bps"].nunique() == 2
    assert len(selected) == 2
    assert set(selected["selection_reason"]).issubset(
        {"fit_strict_path_bar_best", "fit_relative_path_bar_best", "fallback_fit_path_quality_score"}
    )
