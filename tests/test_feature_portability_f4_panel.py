from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_portability_f4_panel import (
    F4CandidatePanelError,
    FINAL_OOS_START,
    materialize_tp6_f4_candidate_panel,
    write_tp6_f4_candidate_panel,
)


@dataclass(frozen=True)
class _Contract:
    cost_bps: float = 100.0


def _source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2023-12-20", tz="UTC")
    for side_index, side in enumerate(("long", "short")):
        for offset in range(100):
            decision = start + pd.Timedelta(hours=offset)
            net = float(offset - 50 + 3 * side_index)
            rows.append({
                "candidate_id": f"asset-{side}-{offset}", "decision_ts": decision,
                "label_available_ts": decision + pd.Timedelta(hours=13), "side_name": side,
                "asset": "asset", "gross_bps": net + 100.0, "net_bps": net,
                "r3_class": 0 if offset < 30 else (1 if offset < 60 else 2),
                "robust_clear_event_b0": 1, "robust_clear_event_b25": 1, "robust_clear_event_b50": 1,
                "long_x": float(offset + side_index), "short_x": float(200 - offset + side_index),
            })
    # It looks like a candidate in the source but must never survive the
    # panel materialiser's end-exclusive November boundary.
    rows.append({
        "candidate_id": "november-final", "decision_ts": FINAL_OOS_START,
        "label_available_ts": FINAL_OOS_START + pd.Timedelta(hours=13), "side_name": "long",
        "asset": "asset", "gross_bps": 110.0, "net_bps": 10.0, "r3_class": 2,
        "robust_clear_event_b0": 1, "robust_clear_event_b25": 1, "robust_clear_event_b50": 1,
        "long_x": 101.0, "short_x": 99.0,
    })
    return pd.DataFrame(rows)


def _loader(**kwargs) -> pd.DataFrame:
    frame = _source()
    assert tuple(kwargs["sides"]) in {("long",), ("short",)}
    start = pd.Timestamp(kwargs["start"])
    end = pd.Timestamp(kwargs["end"])
    return frame.loc[
        frame.side_name.isin(kwargs["sides"]) & frame.decision_ts.ge(start) & frame.decision_ts.lt(end)
    ].copy()


def _features(_: _Contract) -> dict[str, list[str]]:
    return {"long": ["long_x"], "short": ["short_x"]}


def _dispositions() -> pd.DataFrame:
    return pd.DataFrame({"feature": ["long_x", "short_x"], "disposition": ["KEEP_PORTABLE", "INVARIANT_RAW"]})


def _materialized():
    return materialize_tp6_f4_candidate_panel(
        contract=_Contract(), portability_dispositions=_dispositions(),
        load_population=_loader, frozen_features_provider=_features,
    )


def test_materialises_actual_f0_f3_contracts_and_r3_cost_inputs() -> None:
    result = _materialized()
    panel = result.panel
    assert len(panel) == 200
    assert panel.candidate_id.is_unique
    assert panel.decision_ts.lt(FINAL_OOS_START).all()
    assert "november-final" not in set(panel.candidate_id)
    assert np.allclose(panel.gross_bps - panel.net_bps, 100.0)
    assert {"r3_class", "gross_bps", "net_bps", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"}.issubset(panel)
    f0 = result.representation_contracts["F0_current_frozen"]
    f3 = result.representation_contracts["F3_plus_relative"]
    assert f0 == {"long": ["long_x"], "short": ["short_x"]}
    for side, source in (("long", "long_x"), ("short", "short_x")):
        assert f3[side][:1] == [source]
        assert set(f3[side][1:]) == {
            f"{source}__causal_rank_w90", f"{source}__causal_rank_w180",
            f"{source}__causal_robust_z_w90", f"{source}__causal_robust_z_w180",
            f"{source}__causal_delta_p4", f"{source}__causal_delta_p24",
        }
        assert panel.loc[panel.side_name.eq(side), f"{source}__causal_rank_w90"].iloc[29:].notna().all()
    assert result.r3_cost_contract["expected_cost_bps"] == 100.0
    assert result.r3_cost_contract["robust_clear_columns"] == ["robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"]


def test_writes_the_exact_cli_input_contract_files(tmp_path) -> None:
    paths = write_tp6_f4_candidate_panel(_materialized(), tmp_path / "f4_panel")
    assert all(path.exists() for path in paths.values())
    transports = json.loads(paths["transports"].read_text())
    assert [(row["evaluation_start"], row["evaluation_end"]) for row in transports] == [
        ("2024-01-01T00:00:00+00:00", "2024-07-01T00:00:00+00:00"),
        ("2024-07-01T00:00:00+00:00", "2024-11-01T00:00:00+00:00"),
    ]
    model_contract = json.loads(paths["frozen_r3_model_contract"].read_text())
    assert model_contract["model_hpo_performed"] is False


def test_rejects_f3_when_all_actual_f0_sources_are_lineage_rejected() -> None:
    bad = pd.DataFrame({"feature": ["long_x", "short_x"], "disposition": ["REJECTED_LINEAGE", "ERA_SHORTCUT"]})
    with pytest.raises(F4CandidatePanelError, match="no lineage-safe F3 source"):
        materialize_tp6_f4_candidate_panel(
            contract=_Contract(), portability_dispositions=bad,
            load_population=_loader, frozen_features_provider=_features,
        )
