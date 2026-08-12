from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.feature_portability_selection import (
    FeaturePortabilitySelectionError,
    FeaturePortabilitySelectionPolicy,
    select_feature_portability_contract,
    validate_lineage_and_audit,
    write_feature_portability_selection_artifacts,
)
from extreme_price_movements.feature_portability_f4_compact import (
    compact_contract_payload,
    compact_contracts_for_ranked_groups,
    f4_transform_groups,
)


def _evidence() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {"representation": "F1", "transport": "a", "feature_count": 12, "coverage": 1.0, "incremental_top10_net_bps": 7.0, "incremental_top5_net_bps": 2.0, "incremental_worst_month_top10_net_bps": -1.0, "incremental_rank_ic": 0.02, "transport_mda_bps": 3.0, "development_stage": "development_transport", "chronological_verified": True, "global_ranking_verified": True},
            {"representation": "F1", "transport": "b", "feature_count": 12, "coverage": 1.0, "incremental_top10_net_bps": 5.0, "incremental_top5_net_bps": 1.0, "incremental_worst_month_top10_net_bps": -2.0, "incremental_rank_ic": 0.01, "transport_mda_bps": 2.0, "development_stage": "development_transport", "chronological_verified": True, "global_ranking_verified": True},
            {"representation": "F2_compact", "transport": "a", "feature_count": 4, "coverage": 1.0, "incremental_top10_net_bps": 6.6, "incremental_top5_net_bps": 0.5, "incremental_worst_month_top10_net_bps": -3.0, "incremental_rank_ic": 0.01, "transport_mda_bps": 2.5, "development_stage": "development_transport", "chronological_verified": True, "global_ranking_verified": True},
            {"representation": "F2_compact", "transport": "b", "feature_count": 4, "coverage": 1.0, "incremental_top10_net_bps": 5.4, "incremental_top5_net_bps": 0.4, "incremental_worst_month_top10_net_bps": -2.0, "incremental_rank_ic": 0.01, "transport_mda_bps": 2.5, "development_stage": "development_transport", "chronological_verified": True, "global_ranking_verified": True},
        ]
    )
    frame["ranking_scope"] = "pooled_global"
    frame["model_hpo_performed"] = False
    return frame


def _lineage() -> list[dict[str, object]]:
    output = []
    for representation, features in (("F1", ["x", "y"]), ("F2_compact", ["z"])):
        for transport in ("a", "b"):
            output.append({"arm": representation, "run": transport, "feature_count": 12 if representation == "F1" else 4, "features": features, "oof_materialised": True})
    return output


def _audit() -> pd.DataFrame:
    return pd.DataFrame({"feature": ["x", "y", "z"], "coverage": [1.0, 1.0, 1.0], "reference_ready": [True, True, True]})


def test_selects_smallest_representation_within_one_se_and_reports_secondaries() -> None:
    checked = validate_lineage_and_audit(_evidence(), _lineage(), _audit())
    result = select_feature_portability_contract(checked)
    assert result.selected is not None
    assert result.selected["representation"] == "F2_compact"
    row = result.diagnostics.set_index("representation").loc["F1"]
    assert row["stable_transport_mda_score_bps"] == pytest.approx(2.25)
    assert row["top5_incremental_median_bps"] == pytest.approx(1.5)
    assert result.manifest["final_november_oos_consumed"] is False


def test_rejects_missing_transform_audit_and_non_positive_transport_evidence() -> None:
    checked = validate_lineage_and_audit(_evidence(), _lineage(), _audit().loc[lambda value: value.feature.ne("z")])
    result = select_feature_portability_contract(checked)
    rejected = result.diagnostics.set_index("representation").loc["F2_compact"]
    assert not rejected["admissible"]
    assert "audit_missing_declared_features" in rejected["rejection_reasons"]
    bad = _evidence()
    bad.loc[(bad.representation == "F1") & (bad.transport == "b"), "incremental_top10_net_bps"] = 0.0
    result = select_feature_portability_contract(validate_lineage_and_audit(bad, _lineage(), _audit()))
    assert "non_positive_incremental_top10_in_transport" in result.diagnostics.set_index("representation").loc["F1", "rejection_reasons"]


def test_final_oos_and_per_side_or_timestamp_ranking_fail_closed() -> None:
    bad = _evidence()
    bad["period"] = "2024-11"
    with pytest.raises(FeaturePortabilitySelectionError, match="final November"):
        validate_lineage_and_audit(bad, _lineage(), _audit())
    bad = _evidence()
    bad.loc[0, "global_ranking_verified"] = False
    result = select_feature_portability_contract(validate_lineage_and_audit(bad, _lineage(), _audit()))
    assert "not_one_pooled_global_ranking" in result.diagnostics.set_index("representation").loc["F1", "rejection_reasons"]


def test_writer_is_immutable_and_records_input_hashes(tmp_path) -> None:
    checked = validate_lineage_and_audit(_evidence(), _lineage(), _audit())
    result = select_feature_portability_contract(checked, policy=FeaturePortabilitySelectionPolicy(required_transports=("a", "b")))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"source": "test"}), encoding="utf-8")
    output = tmp_path / "f4"
    paths = write_feature_portability_selection_artifacts(result, output, input_paths={"source": source})
    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["inputs"]["source"]["sha256"]
    with pytest.raises(FileExistsError):
        write_feature_portability_selection_artifacts(result, output, input_paths={"source": source})


def test_compact_f4_requires_nonnegative_full_f3_lift_and_writes_exact_manifest(tmp_path) -> None:
    evidence = pd.DataFrame([
        {"representation": "F4_compact_top01", "transport": "a", "feature_count": 2, "coverage": 1.0,
         "incremental_top10_net_bps": 3.0, "incremental_vs_f3_top10_net_bps": 0.5, "transport_mda_bps": 1.0,
         "full_f3_control_eligible": True,
         "development_stage": "development_transport", "chronological_verified": True,
         "global_ranking_verified": True, "ranking_scope": "pooled_global", "model_hpo_performed": False,
         "lineage_audit_verified": True, "lineage_audit_reasons": ""},
        {"representation": "F4_compact_top01", "transport": "b", "feature_count": 2, "coverage": 1.0,
         "incremental_top10_net_bps": 2.0, "incremental_vs_f3_top10_net_bps": -0.01, "transport_mda_bps": 1.0,
         "full_f3_control_eligible": True,
         "development_stage": "development_transport", "chronological_verified": True,
         "global_ranking_verified": True, "ranking_scope": "pooled_global", "model_hpo_performed": False,
         "lineage_audit_verified": True, "lineage_audit_reasons": ""},
    ])
    policy = FeaturePortabilitySelectionPolicy(
        required_transports=("a", "b"), required_representation_prefix="F4_compact_top",
        require_nonnegative_f3_control_lift=True,
    )
    rejected = select_feature_portability_contract(evidence, policy=policy)
    assert rejected.selected is None
    assert "harms_full_f3_control_in_transport" in rejected.diagnostics.iloc[0].rejection_reasons

    evidence.loc[evidence.transport.eq("b"), "incremental_vs_f3_top10_net_bps"] = 0.0
    result = select_feature_portability_contract(evidence, policy=policy)
    assert result.selected is not None
    groups = f4_transform_groups({
        "long": ["x", "x__causal_rank_w90", "x__causal_rank_w180", "x__causal_robust_z_w90", "x__causal_robust_z_w180", "x__causal_delta_p4", "x__causal_delta_p24"],
        "short": ["y", "y__causal_rank_w90", "y__causal_rank_w180", "y__causal_robust_z_w90", "y__causal_robust_z_w180", "y__causal_delta_p4", "y__causal_delta_p24"],
    })
    ranking = ("rank_w90", "rank_w180", "robust_z_w90", "robust_z_w180", "delta_p4", "delta_p24")
    compact = compact_contracts_for_ranked_groups(groups, ranked_transform_groups=ranking)
    compact_payload = compact_contract_payload(
        source_representation="F3_plus_relative", by_transport={"a": compact, "b": compact},
        ranking_by_transport={"a": ranking, "b": ranking},
    )
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    paths = write_feature_portability_selection_artifacts(
        result, tmp_path / "f4", input_paths={"source": source}, compact_contracts=compact_payload,
    )
    payload = json.loads(paths["portable_feature_manifest"].read_text())
    assert payload["status"] == "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST"
    assert payload["full_f3_control_eligible"] is True
    assert payload["feature_contract"] == {"long": ["x", "x__causal_rank_w90"], "short": ["y", "y__causal_rank_w90"]}
    assert payload["selection_artifact"]["path"] == "f4_selected_feature_contract.json"


def test_compact_f4_does_not_require_full_f3_noninferiority_when_f3_is_coverage_ineligible() -> None:
    evidence = pd.DataFrame([
        {"representation": "F4_compact_top01", "transport": transport, "feature_count": 2, "coverage": 1.0,
         "incremental_top10_net_bps": 2.0, "incremental_vs_f3_top10_net_bps": float("nan"),
         "full_f3_control_eligible": False, "transport_mda_bps": 1.0,
         "development_stage": "development_transport", "chronological_verified": True,
         "global_ranking_verified": True, "ranking_scope": "pooled_global", "model_hpo_performed": False,
         "lineage_audit_verified": True, "lineage_audit_reasons": ""}
        for transport in ("a", "b")
    ])
    result = select_feature_portability_contract(
        evidence,
        policy=FeaturePortabilitySelectionPolicy(
            required_transports=("a", "b"), required_representation_prefix="F4_compact_top",
            require_nonnegative_f3_control_lift=True,
        ),
    )
    assert result.selected is not None
    assert result.selected["full_f3_control_eligible"] is False
    assert bool(result.diagnostics.iloc[0]["both_transport_nonnegative_full_f3_lift"])
