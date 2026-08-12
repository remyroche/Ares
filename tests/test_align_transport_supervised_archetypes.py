import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/align_transport_supervised_archetypes.py"
SPEC = importlib.util.spec_from_file_location("align_transport_supervised_archetypes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_consensus_requires_recurrence_and_keeps_soft_definition_discovery_only():
    rows = []
    for fold, threshold in [(2, 0.1), (3, 0.2), (4, 0.3)]:
        rows.append({
            "fold": fold, "side_name": "long", "head": "clear", "conditions": '["x"]',
            "leaf_value": 10.0, "leaf_rows": 100,
        })
        # Substitute a valid JSON rule after exercising the dataframe setup.
        rows[-1]["conditions"] = f'[["x", 1, {threshold}], ["y", -1, 1.0]]'
    rows.append({"fold": 2, "side_name": "long", "head": "clear", "conditions": '[["z", 1, 0.0], ["w", -1, 0.0]]', "leaf_value": 30.0, "leaf_rows": 100})
    alignment, definitions = MODULE.build_consensus(pd.DataFrame(rows), maximum_definitions_per_group=4)
    chosen = alignment.loc[alignment.selected_for_definition]
    assert len(chosen) == 1
    assert chosen.iloc[0].family_signature == "x> & y<"
    assert chosen.iloc[0].n_recurring_folds == 3
    assert chosen.iloc[0].support_eligible == False
    rule = definitions["definitions"]["long"]["clear"][0]
    assert rule["promotion_status"].startswith("DISCOVERY_ONLY")
    assert rule["conditions"][0]["threshold_robust_standard_units"] == 0.2


def test_parse_conditions_tightens_duplicate_directional_bounds():
    parsed = MODULE.parse_conditions('[["x", -1, 2.0], ["x", -1, 1.0], ["x", 1, -3.0]]')
    assert parsed == (("x", -1, 1.0), ("x", 1, -3.0))
