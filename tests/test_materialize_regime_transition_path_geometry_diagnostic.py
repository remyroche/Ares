import json
from pathlib import Path

import pandas as pd

from scripts.materialize_regime_transition_path_geometry_diagnostic import (
    join_path_geometry_context,
    materialize_regime_transition_path_geometry_diagnostic,
    summarize_path_geometry,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    ledger = root / "ledger"
    ledger.mkdir()
    source = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    pd.DataFrame({"source_utc": source, "target__pooled_state": [0, 0, 1, 1, 2, 2]}).to_parquet(
        ledger / "hourly_state_calendar.parquet", index=False
    )
    phase = root / "phases.parquet"
    pd.DataFrame({
        "source_utc": source,
        "target__pattern_phase": ["stable_origin", "approach", "trigger", "active_dislocation", "confirmation", "settled"],
        "target__pattern_phase_available_utc": source + pd.Timedelta(hours=12),
    }).to_parquet(phase, index=False)
    labels = root / "labels.parquet"
    rows = []
    for hour in source[:4]:
        for side in ("long", "short"):
            rows.append({
                "__ts__": hour,
                "__decision_ts__": hour + pd.Timedelta(hours=1),
                "side_name": side,
                "candidate_id": f"{hour.isoformat()}-{side}",
                "__opportunity_occurred_12h__": int(hour.hour % 2 == 0),
                "__peak_mfe_atr_12h__": 2.0 if side == "long" else 1.0,
                "__mae_before_meaningful_mfe_atr_12h__": 0.3,
                "__time_to_first_meaningful_mfe_hours_12h__": 3.0,
                "__future_slope_atr_per_hour_12h__": 0.4,
                "__timeout_outcome_12h__": 0,
                "__exit_conversion_failure_proxy_12h__": 0,
                "__exit_conversion_loss_return_12h__": 0.01,
                "execution_net_ev_12h": 0.02,
            })
    pd.DataFrame(rows).to_parquet(labels, index=False)
    return ledger, phase, labels


def test_source_time_join_does_not_shift_to_execution_time(tmp_path: Path) -> None:
    ledger, phase, labels = _fixture(tmp_path)
    joined, _ = join_path_geometry_context(ledger_dir=ledger, phase_path=phase, label_paths=[labels])
    assert list(joined.sort_values("__ts__")["regime_state"].unique()) == [0, 1]
    assert (joined["__decision_ts__"] - joined["__ts__"]).eq(pd.Timedelta(hours=1)).all()


def test_summary_keeps_regime_and_phase_separate_and_conditions_time_metrics(tmp_path: Path) -> None:
    ledger, phase, labels = _fixture(tmp_path)
    joined, _ = join_path_geometry_context(ledger_dir=ledger, phase_path=phase, label_paths=[labels])
    summary = summarize_path_geometry(joined)
    assert set(summary["taxonomy"]) == {"regime_state_at_decision", "transition_phase_ex_post"}
    time = summary.loc[summary["metric"].eq("time_to_meaningful_mfe_hours")]
    assert time["condition"].eq("opportunity_only").all()
    assert time["n_candidates"].max() == 1


def test_materializes_manifest_with_research_only_nonpromotion_contract(tmp_path: Path) -> None:
    ledger, phase, labels = _fixture(tmp_path)
    output = tmp_path / "output"
    manifest = materialize_regime_transition_path_geometry_diagnostic(
        ledger_dir=ledger, phase_path=phase, label_paths=[labels], output_dir=output
    )
    reported = pd.read_csv(output / "path_geometry_by_context.csv")
    support = pd.read_csv(output / "context_support.csv")
    saved = json.loads((output / "manifest.json").read_text())
    assert manifest["promotion_eligible"] is False
    assert saved["taxonomies"]["joint_table"].startswith("support only")
    assert {"long", "short"}.issubset(reported["side_name"])
    assert "net_ev_12h" in set(reported["metric"])
    assert set(support.columns) == {"side_name", "regime_state", "transition_phase", "candidates", "decision_hours"}
