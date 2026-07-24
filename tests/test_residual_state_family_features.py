from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.residual_state_family_features import (
    ResidualStateFamilyContract,
    fit_definition,
    mechanism_family,
)


def _fixture() -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(13)
    rows = 240
    frame = pd.DataFrame(
        {
            "short_covering_score_market": rng.normal(size=rows),
            "funding_confirmed_long_flush": rng.normal(size=rows),
        }
    )
    target = (
        (frame["short_covering_score_market"] > 0.8)
        & (frame["funding_confirmed_long_flush"] > 0.5)
    ).to_numpy()
    row = {
        "side_name": "short",
        "archetype_policy_key": "short_default_clean_path",
        "base_feature": "short_covering_score_market",
        "gate_feature": "funding_confirmed_long_flush",
        "form": "positive",
        "lift_q25": 1.8,
        "fpr_q75": 0.12,
        "fold_stability": 0.58,
        "adverse_support": 6,
        "status": "validated_production_candidate",
    }
    return frame, target, row


def test_family_mapping_and_contract_round_trip() -> None:
    frame, target, row = _fixture()
    definition = fit_definition(frame, target, row)
    contract = ResidualStateFamilyContract(
        schema_version=1,
        definitions=(definition,),
        source_feature_contract_hash="sha256:source",
        fit_end="2026-06-30T23:00:00+00:00",
    ).with_hash()
    restored = ResidualStateFamilyContract.from_dict(contract.to_dict())
    assert restored == contract
    assert restored.contract_hash.startswith("sha256:")
    assert mechanism_family(row["base_feature"], row["gate_feature"]) == "leverage_rebuild"


def test_transform_is_local_bounded_and_float32() -> None:
    frame, target, row = _fixture()
    definition = fit_definition(frame, target, row)
    contract = ResidualStateFamilyContract(
        schema_version=1,
        definitions=(definition,),
        source_feature_contract_hash="sha256:source",
        fit_end="2026-06-30T23:00:00+00:00",
    ).with_hash()
    side = np.where(np.arange(len(frame)) % 2 == 0, "short", "long")
    archetype = np.repeat("short_default_clean_path", len(frame))
    output = contract.transform(frame, side, archetype)
    assert output.shape[1] == 25
    assert all(dtype == np.dtype("float32") for dtype in output.dtypes)
    assert output.to_numpy().min() >= 0.0
    assert output.to_numpy().max() <= 1.0
    assert output.loc[side == "long"].to_numpy().sum() == 0.0
    assert output.loc[
        side == "short", "residual_state_family_leverage_rebuild_active"
    ].eq(1.0).all()
    assert definition.status == "validated_production_candidate"


def test_contract_hash_rejects_mutation() -> None:
    frame, target, row = _fixture()
    definition = fit_definition(frame, target, row)
    contract = ResidualStateFamilyContract(
        schema_version=1,
        definitions=(definition,),
        source_feature_contract_hash="sha256:source",
        fit_end="2026-06-30T23:00:00+00:00",
    ).with_hash()
    payload = contract.to_dict()
    payload["fit_end"] = "2026-07-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="hash mismatch"):
        ResidualStateFamilyContract.from_dict(payload)
