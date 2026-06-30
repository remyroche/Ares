import json
from pathlib import Path

import pandas as pd

from scripts.audit_size_action_feature_contract import audit_feature_contract


def test_feature_contract_audit_passes_clean_selected_features(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"fold_id": 0, "model": "stage1", "rank": 1, "feature": "strategy_rank_q90"},
            {"fold_id": 0, "model": "stage1", "rank": 2, "feature": "wallet"},
        ]
    ).to_csv(run_dir / "size_action_selected_features.csv", index=False)

    audit = audit_feature_contract(run_dir=run_dir, scorer_bundle_dir=None)

    assert audit["live_feature_contract_clean"] is True
    assert audit["blockers"] == []
    assert audit["checks"][0]["feature_count"] == 2


def test_feature_contract_audit_blocks_forbidden_scorer_features(tmp_path: Path) -> None:
    scorer_dir = tmp_path / "live_scorer"
    scorer_dir.mkdir()
    (scorer_dir / "size_action_live_scorer_manifest.json").write_text(
        json.dumps({"feature_columns": ["strategy_rank_q90", "best_gain"]})
    )
    (scorer_dir / "size_action_live_feature_contract.json").write_text(
        json.dumps({"required_columns": ["wallet", "group_can_bind"]})
    )

    audit = audit_feature_contract(run_dir=None, scorer_bundle_dir=scorer_dir)

    assert audit["live_feature_contract_clean"] is False
    assert "size_action_live_scorer_manifest.json:best_gain" in audit["blockers"]
    assert "size_action_live_feature_contract.json:group_can_bind" in audit["blockers"]
