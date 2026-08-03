from __future__ import annotations

from hashlib import sha256

import pandas as pd
import pytest

from extreme_price_movements.stage_iii_feature_admission import (
    FeatureAdmissionConfig,
    StageIIIFeatureAdmissionError,
    admit_stage_iii_features,
)


def _digest(name: str) -> str:
    return sha256(name.encode()).hexdigest()


def _phantoms() -> pd.DataFrame:
    return pd.DataFrame([
        {"fold_id": fold, "seed": seed, "phantom_mda": value}
        for fold in (1, 2) for seed in (11, 13) for value in (0.01, 0.02, 0.03)
    ])


def _cells() -> pd.DataFrame:
    rows = []
    for group in ("core", "conditional", "local", "unstable", "noise"):
        for fold in (1, 2):
            for seed in (11, 13):
                raw = {"core": .12, "conditional": .00, "local": .12 if fold == 1 and seed == 11 else .00, "unstable": .12, "noise": .00}[group]
                sign = -1 if group == "unstable" and fold == 2 else 1
                rows.append({
                    "feature_group": group, "fold_id": fold, "seed": seed,
                    "train_environment": f"train_{fold}", "test_environment": "latest" if fold == 2 else "early",
                    "within_era_mda": raw, "transport_mda": raw,
                    "false_positive_loss_mda": raw, "effect_sign": sign,
                    "effect_magnitude": .12, "sign_reversal_explained": False,
                    "conditioned_within_era_mda": .12 if group == "conditional" else raw,
                    "conditioned_transport_mda": .12 if group == "conditional" else raw,
                    "conditioned_false_positive_loss_mda": .12 if group == "conditional" else raw,
                    "conditioned_effect_sign": 1 if group == "conditional" else sign,
                    "conditioned_effect_magnitude": .12,
                    "conditioned_sign_reversal_explained": False,
                })
    return pd.DataFrame(rows)


def _features() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature_name": "f_core_b", "feature_group": "core", "feature_order": 20, "coverage": .99, "null_fraction": .01, "finite_fraction": .99, "live_parity": True, "meta_allowed_key": True},
        {"feature_name": "f_core_a", "feature_group": "core", "feature_order": 10, "coverage": .99, "null_fraction": .01, "finite_fraction": .99, "live_parity": True, "meta_allowed_key": True},
        {"feature_name": "f_cond", "feature_group": "conditional", "feature_order": 30, "coverage": .95, "null_fraction": .05, "finite_fraction": .95, "live_parity": True, "meta_allowed_key": True},
        {"feature_name": "f_local", "feature_group": "local", "feature_order": 40, "coverage": .99, "null_fraction": .01, "finite_fraction": .99, "live_parity": True, "meta_allowed_key": True},
        {"feature_name": "f_unstable", "feature_group": "unstable", "feature_order": 50, "coverage": .99, "null_fraction": .01, "finite_fraction": .99, "live_parity": True, "meta_allowed_key": True},
        {"feature_name": "f_noise", "feature_group": "noise", "feature_order": 60, "coverage": .99, "null_fraction": .01, "finite_fraction": .99, "live_parity": True, "meta_allowed_key": True},
    ])


def _run(features: pd.DataFrame | None = None):
    return admit_stage_iii_features(
        _cells(), _phantoms(), _features() if features is None else features,
        source_digests={"mda": _digest("mda"), "phantoms": _digest("phantoms"), "coverage": _digest("coverage")},
        config=FeatureAdmissionConfig(latest_block="latest"),
    )


def test_admits_only_invariant_and_conditioned_groups_in_exact_feature_order() -> None:
    artifact = _run()
    classes = {row["feature_group"]: row["classification"] for row in artifact.group_audit}
    assert classes == {
        "conditional": "REGIME_CONDITIONAL", "core": "INVARIANT_CORE",
        "local": "REGIME_LOCAL_DIAGNOSTIC", "noise": "REDUNDANT", "unstable": "UNSTABLE",
    }
    assert artifact.admitted_feature_groups == ("conditional", "core")
    assert artifact.admitted_ordered_features == ("f_core_a", "f_core_b", "f_cond")
    assert len(artifact.artifact_sha256) == 64
    core = next(row for row in artifact.group_audit if row["feature_group"] == "core")
    assert "transport_mda" in core["raw"]
    assert core["raw"]["positive_cell_fraction"] >= .70


def test_static_coverage_live_parity_and_meta_key_are_fail_closed() -> None:
    feature = _features()
    feature.loc[feature.feature_name.eq("f_cond"), "coverage"] = .89
    feature.loc[feature.feature_name.eq("f_cond"), "null_fraction"] = .11
    artifact = _run(feature)
    assert artifact.admitted_ordered_features == ("f_core_a", "f_core_b")
    assert next(row for row in artifact.group_audit if row["feature_group"] == "conditional")["static_contract_pass"] is False

    feature = _features()
    feature["live_parity"] = feature["live_parity"].astype(object)
    feature.loc[feature.feature_name.eq("f_core_a"), "live_parity"] = "False"
    with pytest.raises(StageIIIFeatureAdmissionError, match="explicit booleans"):
        _run(feature)


def test_fold_local_q95_and_source_digest_are_required() -> None:
    phantom = _phantoms().iloc[:-2]
    with pytest.raises(StageIIIFeatureAdmissionError, match="at least two phantom"):
        admit_stage_iii_features(
            _cells(), phantom, _features(), source_digests={"mda": _digest("mda")},
            config=FeatureAdmissionConfig(latest_block="latest"),
        )
    with pytest.raises(StageIIIFeatureAdmissionError, match="non-placeholder"):
        admit_stage_iii_features(
            _cells(), _phantoms(), _features(), source_digests={"mda": "a" * 64},
            config=FeatureAdmissionConfig(latest_block="latest"),
        )


def test_evidence_digest_is_order_independent() -> None:
    first = _run()
    second = admit_stage_iii_features(
        _cells().sample(frac=1, random_state=7), _phantoms().sample(frac=1, random_state=9),
        _features().sample(frac=1, random_state=11),
        source_digests={"coverage": _digest("coverage"), "phantoms": _digest("phantoms"), "mda": _digest("mda")},
        config=FeatureAdmissionConfig(latest_block="latest"),
    )
    assert first.artifact_sha256 == second.artifact_sha256


def test_latest_block_must_be_preregistered() -> None:
    with pytest.raises(StageIIIFeatureAdmissionError, match="preregistered"):
        admit_stage_iii_features(
            _cells(), _phantoms(), _features(),
            source_digests={"mda": _digest("mda")},
        )
