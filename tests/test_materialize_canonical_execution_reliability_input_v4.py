from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts import materialize_canonical_execution_reliability_input_v4 as v4


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_updated_roles_approves_decile_but_not_duplicate_alias() -> None:
    roles = v4.updated_roles(
        {
            "default_ev_inputs": ["raw_score", v4.SIDE_GROUP_SIZE],
            "target_only_never_features": ["target"],
        }
    )
    assert v4.DECILE in roles["default_ev_inputs"]
    assert v4.DECILE in roles["candidate_context_inputs"]
    assert v4.DECILE_ALIAS not in roles["default_ev_inputs"]
    assert v4.DECILE_ALIAS not in roles["candidate_context_inputs"]
    assert roles["candidate_context_contract"]["duplicate_alias"]["equals"] == v4.SIDE_GROUP_SIZE


def test_updated_roles_rejects_alias_as_feature() -> None:
    try:
        v4.updated_roles(
            {
                "default_ev_inputs": [v4.DECILE_ALIAS],
                "target_only_never_features": [],
            }
        )
    except v4.ReliabilityV4Error as error:
        assert "alias" in str(error)
    else:
        raise AssertionError("alias feature was accepted")


def test_run_copies_panel_byte_identically_and_seals(tmp_path: Path) -> None:
    source = tmp_path / "v3"
    source.mkdir()
    panel = pd.DataFrame(
        {
            v4.DECILE: [0, 3],
            v4.DECILE_ALIAS: [120, 120],
            v4.SIDE_GROUP_SIZE: [120, 120],
        }
    )
    panel.to_parquet(source / "panel.parquet", index=False)
    roles = {"default_ev_inputs": ["raw_score"], "target_only_never_features": ["target"]}
    (source / "feature_roles.json").write_text(json.dumps(roles))
    (source / "capture_support.csv").write_text("metric,rows\nexample,2\n")
    source_outputs = {
        name: _sha(source / name)
        for name in ("panel.parquet", "feature_roles.json", "capture_support.csv")
    }
    source_manifest = {"schema": "canonical_execution_reliability_input_v3", "outputs_sha256": source_outputs}
    (source / "manifest.json").write_text(json.dumps(source_manifest))
    (source / "manifest.sha256").write_text(_sha(source / "manifest.json") + "  manifest.json\n")

    output = tmp_path / "v4"
    manifest = v4.run(type("Args", (), {"source": source, "output_dir": output})())
    assert manifest["byte_identity"]["panel.parquet"] is True
    assert _sha(output / "panel.parquet") == _sha(source / "panel.parquet")
    assert v4.verify(output, "canonical_execution_reliability_input_v4")["rows"] == 2
    output_roles = json.loads((output / "feature_roles.json").read_text())
    assert v4.DECILE in output_roles["default_ev_inputs"]
    assert v4.DECILE_ALIAS not in output_roles["default_ev_inputs"]
