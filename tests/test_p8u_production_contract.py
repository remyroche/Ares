from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_production_contract import (
    P8UPreproductionBundle,
    P8URouterFirstBoundary,
    assert_downstream_is_routed,
    exact_timestamp_route,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> Path:
    router = tmp_path / "router.json"
    base = tmp_path / "base.json"
    under = tmp_path / "under.json"
    router.write_text(json.dumps({"feature_contract": ["r1", "r2"]}))
    base.write_text(json.dumps({"selected_features": ["b1", "r2"]}))
    under.write_text(json.dumps({"selected_features": ["u1", "b1"]}))
    extras = []
    for name in ("canonical", "hpo", "under_score", "mc1", "policy", "portfolio", "feature", "guard"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}" if name == "canonical" else name)
        extras.append((name, path))
    artifacts = {
        "router_contract": {"path": "router.json", "type": "file", "sha256": _hash(router)},
        "base_feature_contract": {"path": "base.json", "type": "file", "sha256": _hash(base)},
        "under_feature_contract": {"path": "under.json", "type": "file", "sha256": _hash(under)},
    }
    for name, path in extras:
        artifacts[name] = {"path": path.name, "type": "file", "sha256": _hash(path)}
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "schema": "strict_r3_p8u_preproduction_bundle_v1",
        "side": "long",
        "routing": {"fraction": .5},
        "runtime": {"order_submission": False, "promotion_status": "blocked_preproduction"},
        "artifacts": artifacts,
    }))
    return bundle


def test_bundle_hash_lock_and_automatic_feature_union(tmp_path: Path) -> None:
    bundle_path = _bundle(tmp_path)
    bundle = P8UPreproductionBundle.load(bundle_path, root=tmp_path)
    assert len(bundle.verify_artifacts()) == 11
    plan = bundle.feature_plan()
    assert plan.router_features == ("r1", "r2")
    assert plan.routed_union == ("b1", "r2", "u1")
    assert plan.full_union == ("r1", "r2", "b1", "u1")
    (tmp_path / "base.json").write_text(json.dumps({"selected_features": ["tampered"]}))
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle.verify_artifacts()


def test_canonical_policy_reference_must_match_the_sealed_policy(tmp_path: Path) -> None:
    bundle_path = _bundle(tmp_path)
    payload = json.loads(bundle_path.read_text())
    payload["artifacts"]["canonical_contract"] = payload["artifacts"].pop("canonical")
    payload["artifacts"]["policy"] = payload["artifacts"].pop("policy")
    canonical_path = tmp_path / "canonical.json"
    policy_path = tmp_path / "policy.json"
    canonical_path.write_text(json.dumps({
        "policy": {"frozen_policy": "policy.json", "frozen_policy_sha256": _hash(policy_path)}
    }))
    payload["artifacts"]["canonical_contract"]["sha256"] = _hash(canonical_path)
    bundle_path.write_text(json.dumps(payload))
    bundle = P8UPreproductionBundle.load(bundle_path, root=tmp_path)
    bundle.verify_artifacts()
    canonical_path.write_text(json.dumps({
        "policy": {"frozen_policy": "policy.json", "frozen_policy_sha256": "wrong"}
    }))
    payload["artifacts"]["canonical_contract"]["sha256"] = _hash(canonical_path)
    bundle_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="policy hash"):
        P8UPreproductionBundle.load(bundle_path, root=tmp_path).verify_artifacts()


def test_router50_is_deterministic_and_downstream_cannot_bypass_it() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["e", "d", "c", "b", "a", "z", "y"],
        "__decision_ts__": ["2026-08-28T00:00:00Z"] * 5 + ["2026-08-28T01:00:00Z"] * 2,
        "side_name": ["long"] * 7,
        "router_score": [1.0, 1.0, .5, .4, .3, 2.0, 1.0],
    })
    routed = exact_timestamp_route(frame, score_column="router_score", fraction=.5)
    # ceil(.5 * 5) = 3; tied 1.0 scores use candidate ID ascending as tie-break.
    assert routed.loc[routed["router50_eligible"], "candidate_id"].tolist() == ["e", "d", "c", "z"]
    accepted = routed.loc[routed["router50_eligible"], ["candidate_id", "__decision_ts__", "side_name"]]
    assert_downstream_is_routed(routed, accepted, layer="Base")
    invalid = pd.concat([accepted, routed.iloc[[4]][["candidate_id", "__decision_ts__", "side_name"]]], ignore_index=True)
    with pytest.raises(ValueError, match="bypass Router50"):
        assert_downstream_is_routed(routed, invalid, layer="MC1")


def test_router_boundary_only_exposes_router50_to_base_under_and_scores(tmp_path: Path) -> None:
    bundle = P8UPreproductionBundle.load(_bundle(tmp_path), root=tmp_path)
    boundary = P8URouterFirstBoundary(bundle.feature_plan())
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": ["2026-08-28T00:00:00Z"] * 4,
        "side_name": ["long"] * 4,
        "r1": [1.0, 2.0, 3.0, 4.0],
        "r2": [0.0] * 4,
        "b1": [0.0] * 4,
        "u1": [0.0] * 4,
    })
    # A scorer is permitted to change row order, but never identities.
    scores = candidates.loc[[3, 1, 2, 0], ["candidate_id", "__decision_ts__", "side_name"]].copy()
    scores["router_score"] = [4.0, 2.0, 3.0, 1.0]
    routed = boundary.gate(candidates, scores)
    base = boundary.downstream_inputs(routed, layer="Base")
    under = boundary.downstream_inputs(routed, layer="Under")
    assert set(base["candidate_id"]) == {"c", "d"}
    assert set(under["candidate_id"]) == {"c", "d"}
    boundary.verify_scored_output(routed, base.assign(base_score=1.0), layer="Base")
    with pytest.raises(ValueError, match="bypass Router50"):
        boundary.verify_scored_output(
            routed,
            candidates.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].assign(base_score=1.0),
            layer="Base",
        )


def test_feature_coverage_fails_closed_until_under_union_is_materialised(tmp_path: Path) -> None:
    bundle = P8UPreproductionBundle.load(_bundle(tmp_path), root=tmp_path)
    partial = bundle.feature_coverage(["r1", "r2", "b1"])
    assert partial.router_missing == ()
    assert partial.base_missing == ()
    assert partial.under_missing == ("u1",)
    with pytest.raises(ValueError, match="does not satisfy"):
        bundle.assert_feature_coverage(["r1", "r2", "b1"])
    assert bundle.assert_feature_coverage(["r1", "r2", "b1", "u1"]).complete
    with pytest.raises(PermissionError, match="research/shadow"):
        bundle.verify_for_submission(["r1", "r2", "b1", "u1"])
