from __future__ import annotations

import pandas as pd
import pytest

from scripts import audit_frozen_side_representation_replay as audit


def _frame(*, start: str, side: str, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    result = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A/USD:USD"] * rows,
            "side_name": [side] * rows,
            "candidate_id": [f"A/USD:USD|{value.isoformat()}|1h|{side}" for value in timestamps],
            "rep": list(range(rows)),
        }
    )
    return result


def test_exact_identity_overlap_never_asofs_adjacent_windows() -> None:
    historical = _frame(start="2026-07-19T15:00:00Z", side="long", rows=1)
    candidates = _frame(start="2026-07-20T00:00:00Z", side="long", rows=1)

    assert audit.exact_identity_overlap(historical, candidates).empty


def test_deterministic_sample_is_side_month_stratified_and_finite() -> None:
    frame = pd.concat(
        [
            _frame(start="2026-04-01T00:00:00Z", side="long", rows=4),
            _frame(start="2026-05-01T00:00:00Z", side="long", rows=4),
            _frame(start="2026-04-01T00:00:00Z", side="short", rows=4),
            _frame(start="2026-05-01T00:00:00Z", side="short", rows=4),
        ],
        ignore_index=True,
    )
    frame.loc[(frame["side_name"] == "short") & (frame["rep"] == 0), "rep"] = float("nan")

    sample = audit.deterministic_side_sample(
        frame, rows_per_side=2, finite_columns=["rep"]
    )

    assert len(sample) == 4
    assert sample["rep"].notna().all()
    assert (
        sample.groupby(["side_name", sample["__ts__"].dt.strftime("%Y-%m")])
        .size()
        .eq(1)
        .all()
    )


def test_report_write_is_atomic_and_refuses_overwrite(tmp_path) -> None:
    report = tmp_path / "audit.json"
    payload = {"schema": "test", "production_output_written": False}

    audit.write_json_report(report, payload)

    assert report.read_text().strip().startswith("{")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.write_json_report(report, payload)
