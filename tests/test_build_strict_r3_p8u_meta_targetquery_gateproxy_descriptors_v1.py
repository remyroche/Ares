from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_strict_r3_p8u_meta_targetquery_gateproxy_descriptors_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("strict_r3_p8u_meta_gateproxy_descriptors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_crossmodel_receipt_exposes_each_trial_without_changing_target_contract(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_p8u_meta_crossmodel_v1",
        "arm": {
            "name": "magnitude_bps__base_band_block28",
            "family": "magnitude",
            "scale": "bps",
            "query": "base_band_block28",
        },
        "meta_feature_contract": "artifacts/frozen_magnitude_120.json",
    }))
    (tmp_path / "correctness_report.json").write_text(json.dumps({"strict_oof": True}))
    pd.DataFrame({
        "trial": ["lgbm_01", "query_02"],
        "model_family": ["lgbm_rank_xendcg", "catboost_queryrmse"],
        "arm": ["magnitude_bps__base_band_block28"] * 2,
    }).to_parquet(tmp_path / "cross_model_summary.parquet", index=False)

    table = MODULE._screen_arms(tmp_path)

    assert table.arm.tolist() == ["lgbm_01", "query_02"]
    assert table.family.tolist() == ["magnitude", "magnitude"]
    assert table.scale.tolist() == ["bps", "bps"]
    assert table["query"].tolist() == ["base_band_block28", "base_band_block28"]
    assert table.source_feature_contract.tolist() == [
        "artifacts/frozen_magnitude_120.json",
        "artifacts/frozen_magnitude_120.json",
    ]
