from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.materialize_failure_first_current_history import (
    COMBINED_FLAG,
    _sha256,
    run,
)


def test_materializer_preserves_oof_and_retires_opened_forward(
    tmp_path: Path,
) -> None:
    timestamp = pd.date_range(
        "2026-07-01", periods=3, freq="h", tz="UTC"
    )
    ledger = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "execution_decision_utc": timestamp,
            "execution_label_end_utc": timestamp + pd.Timedelta(hours=12),
            "evaluation_origin": ["oof", "oof", "forward"],
            "causal_recent_side_isotonic_ev": [0.1, 0.2, 0.3],
            "execution_gross_ev_12h": [0.2, 0.3, 0.4],
            "execution_net_ev_12h": [0.1, 0.2, 0.3],
            "causal_recent_side_isotonic_ev__is_oof": [True, True, False],
            "causal_recent_side_isotonic_ev__is_forward_oos": [
                False,
                False,
                True,
            ],
        }
    )
    state = pd.DataFrame({"candidate_id": ["a", "b", "c"]})
    ledger_path = tmp_path / "ledger.parquet"
    state_path = tmp_path / "state.parquet"
    report_path = tmp_path / "report.json"
    output = tmp_path / "output"
    ledger.to_parquet(ledger_path, index=False)
    state.to_parquet(state_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "status": "CROSS_MODEL_FORWARD_TRANSFER_DIAGNOSTIC",
                "promotion_allowed": False,
                "forward_rows": 1,
                "forward_end": str(timestamp[-1]),
                "source_hashes": {
                    str(ledger_path): _sha256(ledger_path)
                },
            }
        )
    )
    result = run(
        argparse.Namespace(
            ledger=ledger_path,
            state_source=state_path,
            forward_evaluation_report=report_path,
            output_dir=output,
        )
    )
    history = pd.read_parquet(output / "strict_model_oos_history.parquet")
    assert result["rows"] == 3
    assert history[COMBINED_FLAG].all()
    assert history["failure_first_history_role"].value_counts().to_dict() == {
        "outer_oof": 2,
        "retired_resolved_forward_oos": 1,
    }
