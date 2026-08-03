from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import materialize_canonical_execution_reliability_target_pack as target_pack


def _source_frame() -> pd.DataFrame:
    timestamp = pd.Timestamp("2025-03-20T00:00:00Z")
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "side_name": ["long", "short", "long", "short", "long"],
            "__symbol__": ["A", "B", "C", "D", "E"],
            "__ts__": [timestamp] * 5,
            "execution_decision_utc": [timestamp] * 5,
            "execution_label_end_utc": [timestamp + pd.Timedelta(hours=12)] * 5,
            "pre_exit_mfe_return": [0.014, 0.021, 0.003, 0.011, 0.02],
            "execution_cost_return": [0.010, 0.010, 0.010, 0.010, 0.010],
            "execution_net_ev_12h": [0.005, -0.015, -0.020, -0.001, -0.02],
            "execution_exit_class": ["trailing", "trailing", "full_stop", "timeout", "adverse_exit"],
            "target_pre_exit_economic_opportunity": [1, 1, 0, 1, 1],
        }
    )


def test_build_labels_has_exact_targets_masks_and_classes() -> None:
    labels = target_pack.build_labels(_source_frame())
    assert labels["target_pre_exit_opportunity_0bps"].tolist() == [1, 1, 0, 1, 1]
    assert labels["target_pre_exit_opportunity_25bps"].tolist() == [1, 1, 0, 0, 1]
    assert labels["target_pre_exit_opportunity_50bps"].tolist() == [0, 1, 0, 0, 1]
    assert labels["target_successful_deployed_trailing"].tolist() == [1, 0, 0, 0, 0]
    assert labels["target_deployed_hard_adverse"].tolist() == [0, 0, 1, 0, 1]
    assert labels["target_deployed_other_adverse_exit_attribution_only"].tolist() == [0, 0, 0, 0, 1]
    assert labels["target_severe_loss_100bps"].tolist() == [0, 1, 1, 0, 1]
    assert labels.loc[labels.target_severe_loss_100bps.eq(0), "target_conditional_severe_loss_log1p_100bps"].isna().all()
    assert np.isclose(labels.loc[1, "target_conditional_severe_loss_log1p_100bps"], np.log1p(1.5))
    assert labels["target_deployed_exit_economics_class"].astype(str).tolist() == [
        "successful_trailing",
        "trailing_nonpositive",
        "hard_adverse",
        "timeout",
        "hard_adverse",
    ]
    assert labels.label_available_at_utc.eq(labels.execution_label_end_utc).all()


def test_support_ledgers_apply_strict_purge() -> None:
    source = _source_frame()
    source.loc[0, "execution_decision_utc"] = pd.Timestamp("2025-03-19T00:00:00Z")
    source.loc[0, "execution_label_end_utc"] = pd.Timestamp("2025-03-19T12:00:00Z")
    source.loc[1, "execution_decision_utc"] = pd.Timestamp("2025-03-19T00:00:00Z")
    source.loc[1, "execution_label_end_utc"] = pd.Timestamp("2025-03-19T12:00:00Z")
    labels = target_pack.build_labels(source)
    support, classes = target_pack.support_ledgers(
        labels,
        [{"name": "fold", "validation_start_utc": "2025-03-20T00:00:00Z", "validation_end_utc": "2025-03-21T00:00:00Z"}],
    )
    assert not support.empty and not classes.empty
    assert support.loc[(support.split == "train") & (support.side_name == "short"), "rows"].eq(1).all()
    assert support.loc[(support.split == "valid") & (support.side_name == "short"), "rows"].eq(1).all()


def test_roles_forbid_every_target_as_input() -> None:
    roles = target_pack.target_roles()
    assert set(target_pack.target_columns()).issubset(roles["target_only_never_features"])
    assert roles["deployed_exit_targets"]["other_adverse"]["training"].startswith("FORBIDDEN")
    assert "execution_label_end_utc" in roles["availability"]


def test_sealed_verification_rejects_missing_hash(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "labels.parquet").write_bytes(b"payload")
    manifest = {"schema": "test", "outputs_sha256": {"labels.parquet": target_pack.sha256(root / "labels.parquet")}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256").write_text(target_pack.sha256(root / "manifest.json") + "  manifest.json\n")
    assert target_pack.verify_sealed(root, "test")["schema"] == "test"
    (root / "labels.parquet").write_bytes(b"changed")
    try:
        target_pack.verify_sealed(root, "test")
    except target_pack.TargetPackError:
        pass
    else:
        raise AssertionError("changed output passed sealed verification")
