from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_side_geometry_path_quality_selection import select_side_rows  # noqa: E402


def _row(
    side: str,
    arm: str,
    *,
    top_frac: float = 0.10,
    evw: float = 0.72,
    gross: float = 0.006,
    first_touch_bad: float = 0.15,
    mae_before: float = 0.20,
    underwater: float = 8.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "side": side,
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
                f"{period}_gross_ev_weighted_first_touch_precision": evw,
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


def test_select_side_rows_chooses_independently_by_side() -> None:
    candidates = pd.DataFrame(
        [
            _row("long", "long_good", evw=0.72),
            _row("long", "long_bad", evw=0.50),
            _row("short", "short_bad", evw=0.50),
            _row("short", "short_good", evw=0.75),
        ]
    )

    _expanded, selected = select_side_rows(
        candidates,
        costs=[0.01],
        top_fracs=[0.10],
        min_fit_rows=100,
    )

    assert set(selected["side"]) == {"long", "short"}
    assert set(selected["arm"]) == {"long_good", "short_good"}


def test_select_side_rows_recomputes_cost_specific_net() -> None:
    candidates = pd.DataFrame([_row("long", "a", gross=0.006)])

    expanded, selected = select_side_rows(
        candidates,
        costs=[0.0, 0.01],
        top_fracs=[0.10],
        min_fit_rows=100,
    )

    assert sorted(expanded["cost_bps"].unique().tolist()) == [0.0, 100.0]
    assert selected["cost_bps"].nunique() == 2
    assert selected.loc[selected["cost_bps"].eq(0.0), "fit_mean_capture_net"].iloc[0] == 0.006
    assert selected.loc[selected["cost_bps"].eq(100.0), "fit_mean_capture_net"].iloc[0] == -0.004
