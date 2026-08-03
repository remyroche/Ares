import pandas as pd
import pytest
from scripts.run_hourly_transition_semantic_signature_ablation import GROUPS, labelled_hourly


def test_fixed_semantics_have_no_cluster_identifier() -> None:
    assert all("gmm" not in field and "component" not in field and "state" not in field for fields in GROUPS.values() for field in fields)


def test_hourly_onset_label_keeps_event_identity_and_rejects_overlap() -> None:
    panel=pd.DataFrame({"source_utc":pd.date_range("2025-01-01",periods=8,freq="h",tz="UTC")})
    events=pd.DataFrame({"event_id":["event-a"],"anchor_source_utc":[panel.source_utc.iloc[3]]})
    labelled=labelled_hourly(panel,events)
    positive=labelled.loc[labelled.target_onset_next_3h.eq(1)]
    assert set(positive.next_event_id)=={"event-a"}
    overlapping=pd.concat([events,pd.DataFrame({"event_id":["event-b"],"anchor_source_utc":[panel.source_utc.iloc[4]]})],ignore_index=True)
    with pytest.raises(ValueError,match="overlapping"):
        labelled_hourly(panel,overlapping)
