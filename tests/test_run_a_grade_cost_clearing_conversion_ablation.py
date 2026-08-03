import importlib.util
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("conversion",ROOT/"scripts/run_a_grade_cost_clearing_conversion_ablation.py")
MOD=importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(MOD)

def test_global_top_is_not_timestamp_or_side_quota():
    x=pd.DataFrame({"candidate_id":["a","b","c"],"score":[3.,2.,1.]})
    assert MOD.stable_top(x,"score").tolist()==[True,False,False]

def test_mapping_excludes_unresolved_prior_outcomes():
    prior=pd.DataFrame({"execution_label_end_utc":pd.to_datetime(["2026-01-02T00:00Z","2026-01-03T00:00Z"]),"raw_score":[0.,1.],"execution_net_ev_12h":[-.01,.02]})
    mapped,audit=MOD.causal_map(prior,[.5],pd.Timestamp("2026-01-03T00:00Z"))
    assert not audit["map_eligible"] and pd.isna(mapped[0])

def test_exact_contract_rejects_minute_candidate_rows():
    frame=pd.DataFrame({"candidate_id":["a"],"side_name":["long"],"__symbol__":["X"],"__ts__":pd.to_datetime(["2026-01-01T00:01Z"]),"score_residual_expected_ev":[0.],"score_base_alpha":[0.],"execution_gross_ev_12h":[.02],"execution_cost_return":[.01],"execution_net_ev_12h":[.01],"execution_label_end_utc":pd.to_datetime(["2026-01-01T12:01Z"])})
    try: MOD._assert_exact(frame,"test")
    except ValueError as error: assert "hourly" in str(error)
    else: raise AssertionError("minute candidate rows must fail")


def test_checkpointed_single_fold_requires_earlier_chronological_checkpoint(tmp_path, monkeypatch):
    """A restarted later fold cannot silently change its causal map support."""
    monkeypatch.setattr(MOD, "MIN_TRAIN", 1)
    frame=pd.DataFrame({
        "candidate_id":["a","b"], "side_name":["long","long"], "__symbol__":["X","X"],
        "__ts__":pd.to_datetime(["2026-01-01T00:00Z","2026-01-15T00:00Z"]),
        "score_residual_expected_ev":[.1,.2], "score_base_alpha":[.1,.2],
        "execution_gross_ev_12h":[.02,.03], "execution_cost_return":[.01,.01],
        "execution_net_ev_12h":[.01,.02],
        "execution_label_end_utc":pd.to_datetime(["2026-01-01T12:00Z","2026-01-15T12:00Z"]),
        "lineage_id":["test_lineage","test_lineage"], "evidence_grade":["A","A"],
        "regime_columns":["unused","unused"], "transition_columns":["unused","unused"],
    })
    later=MOD.block_starts(frame)[1]
    try:
        MOD.score_lineage(frame, checkpoints=tmp_path, only_block=later)
    except ValueError as error:
        assert "required earlier checkpoint" in str(error)
    else:
        raise AssertionError("later fold must not score without its chronological predecessor")


def test_checkpoint_identity_digest_changes_when_context_intersection_changes():
    base=pd.DataFrame({"candidate_id":["a","b"],"side_name":["long","short"],"__symbol__":["X","Y"],"__ts__":pd.to_datetime(["2026-01-01T00:00Z","2026-01-01T01:00Z"]),"lineage_id":["a","a"]})
    assert MOD.input_identity_sha(base) != MOD.input_identity_sha(base.iloc[:1].copy())


def test_forward_context_arms_fail_closed_when_semantics_are_incompatible(monkeypatch):
    monkeypatch.setattr(MOD, "MIN_STRICT_FORWARD_ROWS", 1)
    monkeypatch.setattr(MOD, "MIN_STRICT_FORWARD_MONTHS", 1)
    monkeypatch.setattr(MOD, "MIN_MAP", 1)
    def make(ts, lineage):
        return pd.DataFrame({
            "candidate_id":[f"{lineage}a",f"{lineage}b"], "side_name":["long","short"], "__symbol__":["X","Y"],
            "__ts__":pd.to_datetime([ts, pd.Timestamp(ts)+pd.Timedelta(hours=1)]),
            "score_residual_expected_ev":[.1,.2], "score_base_alpha":[.1,.2],
            "execution_gross_ev_12h":[.02,.0], "execution_cost_return":[.01,.01], "execution_net_ev_12h":[.01,-.01],
            "execution_label_end_utc":pd.to_datetime([pd.Timestamp(ts)+pd.Timedelta(hours=12),pd.Timestamp(ts)+pd.Timedelta(hours=13)]),
            "lineage_id":[lineage,lineage],"evidence_grade":["A","A"],
            "regime_columns":["incompatible","incompatible"],"transition_columns":["incompatible","incompatible"],
        })
    historical=make("2025-01-01T00:00Z","hist")
    current=make("2026-01-01T00:00Z","current")
    oof=[]
    for arm in ("baseline_residual_ev","hurdle_alpha"):
        part=historical.loc[:, [*MOD.ID,"execution_label_end_utc",MOD.NET]].copy()
        part["arm"]=arm; part["raw_score"]=[.1,.2]; oof.append(part)
    scores,availability=MOD.frozen_2025_to_2026(historical,pd.concat(oof,ignore_index=True),current)
    assert set(scores.arm)=={"baseline_residual_ev","hurdle_alpha"}
    closed=availability.loc[availability.status.eq("fail_closed_noncomparable_2025_2026_context_feature_contract"),"arm"]
    assert set(closed)=={"hurdle_alpha_regime","hurdle_alpha_transition","hurdle_alpha_regime_transition"}
