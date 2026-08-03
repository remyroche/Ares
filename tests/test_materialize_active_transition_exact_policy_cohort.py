import numpy as np
import pandas as pd

from scripts.materialize_active_transition_exact_policy_cohort import materialize_cohort


def _scores() -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-03 05:00", periods=4, freq="h", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "__ts__": timestamp,
                "__symbol__": "AAA/USD:USD",
                "side_name": "long",
                "candidate_id": f"a{index}",
                "causal_recent_isotonic_ev": 0.04 - 0.01 * index,
                "causal_recent_isotonic_ev__is_oof": True,
                "causal_recent_isotonic_ev__is_forward_oos": False,
            }
        )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    rows = []
    for index, timestamp in enumerate(pd.date_range("2026-06-03 05:00", periods=4, freq="h", tz="UTC")):
        rows.append(
            {
                "__ts__": timestamp,
                "__symbol__": "AAA/USD:USD",
                "side_name": "long",
                "candidate_id": f"a{index}",
                "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
                "execution_label_end_utc": timestamp + pd.Timedelta(hours=13),
                "execution_exit_hour": 2.5,
                "execution_exit_reason": "trailing",
                "execution_entry_price": 100.0,
                "execution_exit_price": 104.0,
                "execution_gross_ev_12h": 0.04,
                "execution_cost_return": 0.01,
                "execution_net_ev_12h": 0.03,
            }
        )
    return pd.DataFrame(rows)


def _active() -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-03 05:00", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "source_utc": timestamps,
            "target__event_id": ["event", None, None, None],
            "target__transition_active": [1, 0, 0, 0],
            "prediction": [0.8, 0.1, 0.1, 0.1],
        }
    )


def test_materializes_actual_exit_path_and_machine_gate() -> None:
    cohort, active, gate = materialize_cohort(_scores(), _labels(), _active())
    assert len(cohort) == 4
    assert len(active) == 1
    first = cohort.loc[cohort["candidate_id"].eq("a0")].iloc[0]
    assert first["entry_price"] == 100.0
    assert first["exit_price"] == 104.0
    assert first["net_return"] == 0.03
    assert first["exit_timestamp"] == pd.Timestamp("2026-06-03 08:30", tz="UTC")
    assert np.isclose(first["holding_bars"], 10.0)
    assert gate["validity"]["exact_execution_lineage"] is True
    assert gate["validity"]["promotion_valid"] is False
    assert gate["coverage"]["active_transition_events"] == 1
    assert gate["validity"]["policy_sweep_economic_effectiveness_informative"] is False


def test_rejects_active_probability_outside_unit_interval() -> None:
    active = _active()
    active.loc[0, "prediction"] = 1.1
    try:
        materialize_cohort(_scores(), _labels(), active)
    except ValueError as exc:
        assert "within [0, 1]" in str(exc)
    else:
        raise AssertionError("out-of-range active probability must fail")
