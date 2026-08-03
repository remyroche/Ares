from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pandas as pd

RUNNER=Path(__file__).resolve().parents[1]/"scripts"/"summarize_canonical_execution_reliability_ablation_v2.py"
SPEC=importlib.util.spec_from_file_location("reliability_summary",RUNNER); assert SPEC and SPEC.loader
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)

def test_summary_reproduces_frozen_path_and_control_gate():
    arms,heads,gates,decision=M.summarize(M.SOURCE,M.CONFIG)
    assert len(arms)==21
    assert decision["support_winner"]=="A1__score4+support_S0+support_S1B"
    assert decision["context_winner"]=="A1_context__timestamp_side_relative"
    assert decision["target_architecture_winner"]=="A2__context__timestamp_side_relative"
    assert decision["final_frozen_challenger"]=="A5__A2__context__timestamp_side_relative"
    assert not decision["promotion_eligible"]
    assert not gates["pass"].all()
    assert not gates.set_index("gate").loc["beats_residual_control_objective","pass"]

def test_summary_contains_latest_period_and_both_sides():
    arms,_,gates,_=M.summarize(M.SOURCE,M.CONFIG)
    final=arms.set_index("config").loc["A5__A2__context__timestamp_side_relative"]
    assert final["april_latest_7d_mapped_top10_net_bps"]<0
    assert final["march_long_net_contribution_bps"]<0
    assert final["march_short_net_contribution_bps"]<0
    assert gates.set_index("gate").loc["all_selection_folds_tie_safe","pass"] in (False,0)
