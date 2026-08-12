from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "score_strict_r3_forward.py"
SPEC = importlib.util.spec_from_file_location("score_strict_r3_forward", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_current_reference_is_rescored_by_held_upstream_bundle(monkeypatch) -> None:
    """The forward entry point delegates each held month to one producer."""
    selected_month: list[str] = []

    class _Bundle:
        manifest = {"bundle_sha256": "held-august"}

    class _Conversion:
        cutoff = pd.Timestamp("2026-07-16T00:00:00Z")

    monkeypatch.setattr(
        MODULE,
        "_monthly_bundle_dir",
        lambda _root, month: selected_month.append(month) or Path("/bundle"),
    )
    monkeypatch.setattr(MODULE, "load_monthly_upstream_bundle", lambda _path: _Bundle())

    seen: list[dict[str, str]] = []

    def _score_by_vintage(bundle, *, reference, held, upstream_bundles):
        assert bundle is conversion
        seen.append({
            "reference": str(len(reference)),
            "held": str(len(held)),
            "upstream": str(upstream_bundles["2026-08"].manifest["bundle_sha256"]),
        })
        output = pd.concat([
            reference.assign(__score_role__="reference"),
            held.assign(__score_role__="held"),
        ], ignore_index=True)
        output["cdf_reference_upstream_bundle_sha256"] = "held-august"
        return output, pd.DataFrame([{
            "same_upstream_bundle_for_reference_and_held": True,
        }])

    monkeypatch.setattr(MODULE, "score_four_week_conversion_by_upstream_vintage", _score_by_vintage)
    reference = pd.DataFrame({
        "candidate_id": ["june", "july"],
        "__decision_ts__": pd.to_datetime(
            ["2026-06-20T00:00:00Z", "2026-07-31T00:00:00Z"], utc=True,
        ),
        "side_name": ["long", "long"],
    })
    held = pd.DataFrame({
        "candidate_id": ["august"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
    })
    conversion = _Conversion()

    scored, audit, hashes = MODULE._score_current_by_upstream_vintage(
        conversion_bundle=conversion,
        reference=reference,
        held=held,
        upstream_root=Path("/root"),
    )

    assert selected_month == ["2026-08"]
    assert seen == [{"reference": "2", "held": "1", "upstream": "held-august"}]
    assert hashes == {"2026-08": "held-august"}
    assert set(scored["cdf_reference_upstream_bundle_sha256"]) == {"held-august"}
    assert audit["same_upstream_bundle_for_reference_and_held"].tolist() == [True]


def test_lockstep_forward_uses_one_persisted_upstream_for_reserve_and_held(monkeypatch) -> None:
    class _Conversion:
        cutoff = pd.Timestamp("2026-07-16T00:00:00Z")
        end_exclusive = pd.Timestamp("2026-08-13T00:00:00Z")

    class _Upstream:
        cutoff = pd.Timestamp("2026-07-16T00:00:00Z")
        end_exclusive = pd.Timestamp("2026-08-13T00:00:00Z")
        manifest = {"bundle_sha256": "one-lockstep-upstream"}

    conversion, upstream = _Conversion(), _Upstream()
    calls: list[tuple[bool, object]] = []

    def _score_upstream(bundle, frame, *, allow_prior_reference=False, prior_reference_start=None):
        assert bundle is upstream
        calls.append((allow_prior_reference, prior_reference_start))
        output = frame[["candidate_id", "__decision_ts__", "side_name"]].copy()
        for column in (
            "base_score", "base_rank42", "base_anchor_bps", "conditional_consensus_rank",
            "upstream", "ordinary_shadow_consensus_rank", "ordinary_shadow_upstream",
        ):
            output[column] = 0.5
        output["upstream_bundle_sha256"] = "one-lockstep-upstream"
        return output

    def _score_conversion(bundle, *, reference, held, chunk_hours):
        assert bundle is conversion
        assert chunk_hours == 24
        assert set(reference["upstream_bundle_sha256"]) == {"one-lockstep-upstream"}
        assert set(held["upstream_bundle_sha256"]) == {"one-lockstep-upstream"}
        return pd.concat([
            reference.assign(__score_role__="reference"),
            held.assign(__score_role__="held"),
        ], ignore_index=True), pd.DataFrame([{"held_percentile_operations": 0}])

    monkeypatch.setattr(MODULE, "score_monthly_upstream_bundle", _score_upstream)
    monkeypatch.setattr(MODULE, "score_four_week_conversion_bundle_lockstep", _score_conversion)
    reference = pd.DataFrame({
        "candidate_id": ["reserve"],
        "__decision_ts__": pd.to_datetime(["2026-07-01T00:00:00Z"], utc=True),
        "side_name": ["long"],
    })
    held = pd.DataFrame({
        "candidate_id": ["held"],
        "__decision_ts__": pd.to_datetime(["2026-07-16T00:00:00Z"], utc=True),
        "side_name": ["long"],
    })

    scored, audit, hashes = MODULE._score_current_lockstep(
        conversion_bundle=conversion,
        upstream_bundle=upstream,
        reference=reference,
        held=held,
        chunk_hours=24,
    )

    assert calls == [
        (True, pd.Timestamp("2026-06-04T00:00:00Z")),
        (False, None),
    ]
    assert hashes == {"lockstep": "one-lockstep-upstream"}
    assert audit["lockstep_producer"].tolist() == [True]
    assert scored["candidate_id"].tolist() == ["reserve", "held"]


def test_frozen_base_availability_requires_every_declared_raw_field() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["complete", "missing", "infinite"],
        "__decision_ts__": pd.to_datetime(["2026-08-01T00:00:00Z"] * 3, utc=True),
        "side_name": ["long"] * 3,
        "a": [1.0, None, 1.0],
        "b": [2.0, 2.0, float("inf")],
    })

    output = MODULE._frozen_base_availability(frame, ("a", "b"))

    assert output["frozen_base_feature_count"].tolist() == [2, 1, 1]
    assert output["frozen_base_contract_complete"].tolist() == [True, False, False]
