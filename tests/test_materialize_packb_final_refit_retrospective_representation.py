from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.base_candidate_population import deterministic_candidate_ids
from scripts import materialize_packb_final_refit_retrospective_representation as adapter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_context(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "packb"
    root.mkdir()
    candidate_source = tmp_path / "candidate_features.parquet"
    pd.DataFrame({"candidate_id": ["source"]}).to_parquet(candidate_source)
    context = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-07-20T00:00:00Z",
                    "2026-07-20T01:00:00Z",
                    "2026-07-20T00:00:00Z",
                    "2026-07-20T01:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["A", "B", "A", "B"],
            "side_name": ["long", "long", "short", "short"],
            "side": [1.0, 1.0, -1.0, -1.0],
            "selected_top40": [True, True, True, True],
            "prediction_source": ["frozen_final_refit"] * 4,
            "execution_decision_utc": pd.to_datetime(
                [
                    "2026-07-20T01:00:00Z",
                    "2026-07-20T02:00:00Z",
                    "2026-07-20T01:00:00Z",
                    "2026-07-20T02:00:00Z",
                ],
                utc=True,
            ),
            "feature_available_at": pd.to_datetime(
                [
                    "2026-07-20T00:00:00Z",
                    "2026-07-20T01:00:00Z",
                    "2026-07-20T00:00:00Z",
                    "2026-07-20T01:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    context["candidate_id"] = deterministic_candidate_ids(context, timeframe="1h")
    context_path = root / "packb_forward_context.parquet"
    context.to_parquet(context_path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "packb_final_refits_forward_v1",
                "status": "frozen_final_refit_preentry_context_not_oos_metrics",
                "contract": {"outcomes_used": False},
                "inputs": {
                    "candidate_features": {
                        "path": str(candidate_source),
                        "sha256": _sha256(candidate_source),
                    }
                },
                "output": {"sha256": _sha256(context_path)},
            }
        )
    )
    return root, candidate_source


def _write_ae_root(tmp_path: Path) -> Path:
    root = tmp_path / "ae"
    sides: dict[str, object] = {}
    for side, payload in (("long", b"long-state"), ("short", b"short-state")):
        state = root / side / "ae_gmm" / "ae_gmm_state.pkl"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(payload)
        sides[side] = {
            "ae_gmm": {
                "status": "FROZEN_SIDE_LOCAL_AE_GMM_STATE",
                "side": side,
                "state_path": str(state),
                "state_sha256": _sha256(state),
            }
        }
    (root / "summary.json").write_text(
        json.dumps({"status": "FROZEN_LONG_AND_SHORT_AE_GMM", "sides": sides})
    )
    return root


def _write_prescore_candidates(tmp_path: Path) -> tuple[Path, Path]:
    context_root, _ = _write_context(tmp_path)
    candidates_root = tmp_path / "candidates"
    candidates_root.mkdir()
    frame = pd.read_parquet(context_root / "packb_forward_context.parquet").drop(
        columns=["selected_top40", "prediction_source"]
    )
    candidates_path = candidates_root / "candidate_features.parquet"
    frame.to_parquet(candidates_path)
    manifest_path = candidates_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "execution_ev_july_retrospective_candidate_surface_v1",
                "status": "materialized_retrospective_non_promotable",
                "outcomes_used": False,
                "candidates_written": True,
                "output": {
                    "sha256": _sha256(candidates_path),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                },
            }
        )
    )
    return candidates_path, manifest_path


class _Guard:
    def __init__(self, *, limits, **_kwargs) -> None:
        self.limits = limits

    def preflight(self, _stage: str) -> None:
        return None

    def checkpoint(self, _stage: str) -> None:
        return None


def _fake_loader(*, side: str, **_kwargs):
    def load(ledger: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
        assert fields == list(adapter.FROZEN_EXECUTION_EV_GENERATED)
        sign = 1.0 if side == "long" else -1.0
        return pd.DataFrame({
            feature: np.full(len(ledger), sign * (index + 1), dtype=np.float32)
            for index, feature in enumerate(fields)
        })

    generated = list(adapter.FROZEN_EXECUTION_EV_GENERATED)
    return load, ["raw_only", *generated], {
        "raw_candidate_features": 1,
        "generated_candidate_features": len(generated),
        "ae_state_sha256": f"{side}-state",
    }


def test_appends_complete_side_local_representation_with_manifest_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_root, _ = _write_context(tmp_path)
    ae_root = _write_ae_root(tmp_path)
    monkeypatch.setattr(adapter, "TrainingResourceGuard", _Guard)
    monkeypatch.setattr(adapter, "_side_loader", _fake_loader)

    destination = tmp_path / "output"
    result = adapter.run(
        context_root=context_root,
        ae_root=ae_root,
        feature_store=tmp_path / "features",
        destination=destination,
    )

    frame = pd.read_parquet(destination / "packb_forward_context_with_representation.parquet")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert frame["candidate_id"].astype(str).tolist() == deterministic_candidate_ids(
        frame, timeframe="1h"
    ).astype(str).tolist()
    assert frame["gmm_representation_available"].eq(1.0).all()
    assert frame[list(adapter.FROZEN_EXECUTION_EV_REPRESENTATION)].notna().all().all()
    assert manifest["outcomes_used"] is False
    assert set(adapter.FROZEN_EXECUTION_EV_GENERATED).issubset(
        result["representation"]["generated_features"]
    )


def test_rejects_changed_candidate_source_provenance(tmp_path: Path) -> None:
    context_root, candidate_source = _write_context(tmp_path)
    candidate_source.write_bytes(b"changed")
    with pytest.raises(adapter.RetrospectiveRepresentationError, match="provenance"):
        adapter._validate_context(context_root=context_root)


def test_prescore_mode_materializes_candidate_ledger_for_packb_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates_path, source_manifest_path = _write_prescore_candidates(tmp_path)
    ae_root = _write_ae_root(tmp_path)
    monkeypatch.setattr(adapter, "TrainingResourceGuard", _Guard)
    monkeypatch.setattr(adapter, "_embedded_candidate_side_loader", lambda **kwargs: _fake_loader(side=kwargs["side"]))

    destination = tmp_path / "prescore_output"
    result = adapter.run(
        candidate_features=candidates_path,
        source_manifest_path=source_manifest_path,
        ae_root=ae_root,
        feature_store=tmp_path / "features",
        destination=destination,
    )

    frame = pd.read_parquet(destination / "candidate_features_with_representation.parquet")
    assert frame["candidate_id"].astype(str).tolist() == deterministic_candidate_ids(
        frame, timeframe="1h"
    ).astype(str).tolist()
    assert set(adapter.FROZEN_EXECUTION_EV_REPRESENTATION).issubset(frame.columns)
    assert frame[list(adapter.FROZEN_EXECUTION_EV_REPRESENTATION)].notna().all().all()
    assert result["input"]["kind"] == "pre_score_candidate_surface"
    assert result["outcomes_used"] is False


def test_frozen_registry_replay_rejects_changed_raw_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_root = (
        Path(__file__).resolve().parents[1]
        / "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/loader_evidence"
    )
    contract = json.loads((loader_root / "frozen_feature_contract.json").read_text())
    evidence = json.loads((loader_root / "loader_evidence.json").read_text())
    monkeypatch.setattr(
        adapter,
        "_provenance_backed_raw_allowlist",
        lambda: (frozenset(), {}, "0" * 64, "1" * 64),
    )
    with pytest.raises(adapter.RetrospectiveRepresentationError, match="allowlist"):
        adapter._validate_frozen_registry_replay_binding(
            loader_root=loader_root,
            contract=contract,
            evidence=evidence,
        )


def test_embedded_prescore_loader_requires_complete_frozen_raw_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_root, _ = _write_context(tmp_path)
    context = pd.read_parquet(context_root / "packb_forward_context.parquet").drop(
        columns=["selected_top40", "prediction_source"]
    )
    ae_root = _write_ae_root(tmp_path)

    class _Contract:
        feature_columns = ("required_raw",)

    class _Frozen:
        @staticmethod
        def from_mapping(_value):
            return _Contract()

    monkeypatch.setattr(adapter, "FrozenFeatureContract", _Frozen)
    monkeypatch.setattr(
        adapter,
        "_load_loader_contract",
        lambda *_args, **_kwargs: ({}, object(), {}),
    )
    with pytest.raises(adapter.RetrospectiveRepresentationError, match="lacks the complete"):
        adapter._embedded_candidate_side_loader(
            side="long", candidate_context=context, ae_root=ae_root
        )


def test_rejects_nonfinite_generated_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_root, _ = _write_context(tmp_path)
    ae_root = _write_ae_root(tmp_path)
    monkeypatch.setattr(adapter, "TrainingResourceGuard", _Guard)

    def nonfinite_loader(**kwargs):
        loader, candidates, evidence = _fake_loader(**kwargs)

        def with_nan(ledger: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
            frame = loader(ledger, fields)
            frame.loc[0, "gmm_ood_score"] = np.nan
            return frame

        return with_nan, candidates, evidence

    monkeypatch.setattr(adapter, "_side_loader", nonfinite_loader)
    with pytest.raises(adapter.RetrospectiveRepresentationError, match="non-finite"):
        adapter.run(
            context_root=context_root,
            ae_root=ae_root,
            feature_store=tmp_path / "features",
            destination=tmp_path / "blocked",
        )
    assert not (tmp_path / "blocked").exists()
