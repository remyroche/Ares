from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements import packb_side_local_ae_stage as ae_stage
from extreme_price_movements import packb_side_stage_manifest as stage_manifest


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


BASE_SOURCE_HASHES = {
    "dec09_decisions_sha256": _sha("dec09"),
    "canonical_shard_inventory_sha256": _sha("inventory"),
    "causal_audit_sha256": _sha("audit"),
    "population_preflight_sha256": _sha("preflight"),
    "feature_store_inventory_sha256": _sha("feature-store"),
    "feature_store_inventory_evidence_sha256": _sha("feature-store-evidence"),
}
SOURCE_REVISION = "a" * 40
CALENDAR_SHA256 = _sha("locked-calendar")


class _RecordingGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def preflight(self, stage: str) -> None:
        self.calls.append(("preflight", stage))

    def checkpoint(self, stage: str) -> None:
        self.calls.append(("checkpoint", stage))


def _row(candidate_id: str, signal: str, *, side: str = "long") -> dict[str, object]:
    signal_ts = pd.Timestamp(signal)
    decision = signal_ts + pd.Timedelta(hours=1)
    return {
        "candidate_id": candidate_id,
        "side_name": side,
        "__ts__": signal_ts,
        "__decision_ts__": decision,
        "__label_resolution_ts__": decision + pd.Timedelta(hours=24),
        "__symbol__": f"SYM_{candidate_id}",
    }


def _cohort(*, include_after_reference: bool = True) -> pd.DataFrame:
    rows = [
        _row("ref-begin", "2025-01-01T00:00:00Z"),
        _row("ref-middle", "2025-06-01T00:00:00Z"),
        _row("ref-end", "2025-10-30T22:00:00Z"),
    ]
    if include_after_reference:
        rows.extend(
            [
                _row("reference-end-excluded", "2025-11-01T00:00:00Z"),
                _row("resolution-end-excluded", "2025-10-30T23:00:00Z"),
                _row("post-reference-excluded", "2026-02-01T00:00:00Z"),
            ]
        )
    return pd.DataFrame(rows)


def _write_ledgers(
    tmp_path: Path, cohort: pd.DataFrame
) -> tuple[Path, Path, dict[str, str]]:
    cohort_path = tmp_path / "cohort.parquet"
    population_path = tmp_path / "authorized_population.parquet"
    cohort.to_parquet(cohort_path, index=False)
    cohort.to_parquet(population_path, index=False)
    return (
        cohort_path,
        population_path,
        {
            **BASE_SOURCE_HASHES,
            "authorized_population_ledger_sha256": stage_manifest.sha256_file(
                population_path
            ),
        },
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger: pd.DataFrame | None = None,
    loader=None,
    fake_state: dict[str, object] | None = None,
    published_output_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], _RecordingGuard]:
    seen: dict[str, object] = {}
    guard = _RecordingGuard()

    def fake_fit(matrix: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        seen["fit_matrix"] = matrix.copy()
        seen["fit_kwargs"] = kwargs
        return fake_state or {"enabled": True, "feature_columns": list(matrix.columns)}

    def default_loader(sampled: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        seen["loader_ledger"] = sampled.copy()
        seen["loader_columns"] = list(columns)
        return pd.DataFrame(
            {
                "observable_a": range(len(sampled)),
                "observable_b": [float(index) for index in range(len(sampled))],
            }
        )

    active_loader = default_loader if loader is None else loader
    loader_evidence_path = tmp_path / "loader_evidence.json"
    loader_hashes = {
        "raw_universe_sha256": _sha("raw-universe"),
        "coverage_profile_sha256": _sha("coverage"),
        "feature_contract_sha256": _sha("feature-contract"),
        "loader_contract_sha256": _sha("loader-contract"),
        "loader_module_sha256": _sha("loader-module"),
        "source_schema_sha256": _sha("source-schema"),
        "source_revision": SOURCE_REVISION,
    }
    loader_evidence_path.write_text(
        json.dumps(loader_hashes, sort_keys=True) + "\n", encoding="utf-8"
    )
    active_loader.packb_static_feature_loader_evidence = {
        **loader_hashes,
        "evidence_path": str(loader_evidence_path),
        "requested_feature_policy": "unique_ordered_subset_of_frozen_contract",
    }
    active_loader.packb_static_feature_contract = {
        "feature_columns": ["observable_a", "observable_b"],
        "feature_contract_sha256": loader_hashes["feature_contract_sha256"],
    }
    active_loader.packb_static_feature_matrix_sha256 = (
        lambda sampled, matrix: hashlib.sha256(
            (
                "|".join(sampled["candidate_id"].astype(str))
                + matrix.to_json(orient="split")
            ).encode("utf-8")
        ).hexdigest()
    )

    monkeypatch.setattr(ae_stage, "fit_ae_gmm_state", fake_fit)
    active_ledger = _cohort() if ledger is None else ledger
    cohort_path, population_path, source_hashes = _write_ledgers(
        tmp_path, active_ledger
    )
    seen["source_hashes"] = source_hashes
    report = ae_stage.fit_side_local_ae_gmm_stage(
        side="long",
        cohort_ledger=active_ledger,
        cohort_ledger_path=cohort_path,
        authorized_population_ledger_path=population_path,
        feature_loader=active_loader,
        input_features=["observable_a", "observable_b"],
        output_dir=tmp_path / "out",
        published_output_dir=published_output_dir,
        source_hashes=source_hashes,
        source_revision=SOURCE_REVISION,
        fixed_calendar_sha256=CALENDAR_SHA256,
        max_train_rows=3,
        gmm_max_train_rows=3,
        min_reference_rows=3,
        resource_guard=guard,
    )
    return report, seen, guard


def test_published_paths_survive_atomic_directory_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = tmp_path / "published"
    report, _seen, _guard = _run(
        tmp_path,
        monkeypatch,
        published_output_dir=published,
    )
    (tmp_path / "out").rename(published)

    for key in (
        "state_path",
        "metadata_path",
        "side_stage_manifest_path",
        "candidate_stream_evidence_path",
    ):
        assert Path(str(report[key])).is_file()
        assert str(report[key]).startswith(str(published))
    metadata = json.loads(
        Path(str(report["metadata_path"])).read_text(encoding="utf-8")
    )
    for key in ("state_path", "stage_config_path"):
        assert Path(str(metadata[key])).is_file()
        assert str(metadata[key]).startswith(str(published))
    candidate = metadata["candidate_stream_evidence"]["path"]
    assert Path(str(candidate)).is_file()
    assert str(candidate).startswith(str(published))


def test_fits_only_authorized_one_side_pre_nov_rows_and_emits_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, seen, guard = _run(tmp_path, monkeypatch)

    loader_ledger = seen["loader_ledger"]
    assert set(loader_ledger["candidate_id"]) == {
        "ref-begin",
        "ref-middle",
        "ref-end",
    }
    assert loader_ledger["side_name"].eq("long").all()
    assert (
        pd.to_datetime(loader_ledger["__ts__"], utc=True)
        .lt(ae_stage.AE_REFERENCE_END_UTC)
        .all()
    )
    assert (
        pd.to_datetime(loader_ledger["__label_resolution_ts__"], utc=True)
        .lt(ae_stage.AE_REFERENCE_END_UTC)
        .all()
    )
    assert list(seen["fit_matrix"].columns) == ["observable_a", "observable_b"]
    kwargs = seen["fit_kwargs"]
    assert kwargs["economic_targets"] == {}
    assert kwargs["require_both_sides"] is False
    assert kwargs["outcome_free"] is True
    assert kwargs["path_aware_hpo"] is False
    assert kwargs["temporal_feature_contract"] == "row_independent_v1"
    assert any(event == "preflight" for event, _stage in guard.calls)
    assert any("before_feature_load" in stage for _event, stage in guard.calls)
    assert any("before_fit" in stage for _event, stage in guard.calls)

    manifest_path = Path(str(report["side_stage_manifest_path"]))
    evidence = stage_manifest.validate_side_stage_manifest(
        manifest_path,
        expected_side="long",
        expected_stage="ae_gmm",
        expected_source_hashes=seen["source_hashes"],
        expected_fixed_calendar_sha256=CALENDAR_SHA256,
    )
    assert evidence["candidate_stream"]["count"] == 3
    assert report["reference_rows_available"] == 3
    assert report["reference_rows_sampled"] == 3
    assert len(report["feature_matrix_sha256"]) == 64
    assert report["feature_loader_evidence_sha256"] == stage_manifest.sha256_file(
        tmp_path / "loader_evidence.json"
    )
    with Path(str(report["state_path"])).open("rb") as handle:
        state = pickle.load(handle)
    assert state["packb_side_scope"] == "long"
    assert state["representation_selection_outcome_keys"] == []
    assert (
        state["cycle_reference_sample_identity_hash"]
        == report["sample_identity_sha256"]
    )


def test_rejects_a_mixed_side_ledger_before_loader_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def never_loader(_sampled: pd.DataFrame, _columns: list[str]) -> pd.DataFrame:
        calls.append("loader")
        raise AssertionError("loader must not be called")

    def never_fit(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("fit")
        raise AssertionError("fit must not be called")

    ledger = pd.concat(
        [
            _cohort(include_after_reference=False),
            pd.DataFrame([_row("short", "2025-02-01T00:00:00Z", side="short")]),
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(ae_stage, "fit_ae_gmm_state", never_fit)
    cohort_path, population_path, source_hashes = _write_ledgers(tmp_path, ledger)

    with pytest.raises(
        ae_stage.PackBSideLocalAEStageError, match="exactly 'long' rows"
    ):
        ae_stage.fit_side_local_ae_gmm_stage(
            side="long",
            cohort_ledger=ledger,
            cohort_ledger_path=cohort_path,
            authorized_population_ledger_path=population_path,
            feature_loader=never_loader,
            input_features=["observable_a", "observable_b"],
            output_dir=tmp_path / "out",
            source_hashes=source_hashes,
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            min_reference_rows=3,
            resource_guard=_RecordingGuard(),
        )
    assert calls == []


def test_rejects_resolution_at_cutoff_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _cohort(include_after_reference=False)
    boundary = _row("cutoff", "2026-02-27T23:00:00Z")
    ledger = pd.concat([ledger, pd.DataFrame([boundary])], ignore_index=True)
    seen: list[str] = []

    def never_loader(_sampled: pd.DataFrame, _columns: list[str]) -> pd.DataFrame:
        seen.append("loader")
        raise AssertionError("loader must not be called")

    with pytest.raises(
        ae_stage.PackBSideLocalAEStageError, match="at/after the pre-March cutoff"
    ):
        _run(tmp_path, monkeypatch, ledger=ledger, loader=never_loader)
    assert seen == []


def test_rejects_cohort_when_bound_ledger_file_has_different_feature_join_identity(
    tmp_path: Path,
) -> None:
    cohort = _cohort(include_after_reference=False)
    cohort_path, population_path, source_hashes = _write_ledgers(tmp_path, cohort)
    bound = cohort.copy()
    bound.loc[0, "__symbol__"] = "CHANGED_ON_DISK"
    bound.to_parquet(cohort_path, index=False)
    calls: list[str] = []

    with pytest.raises(
        ae_stage.PackBSideLocalAEStageError,
        match="does not match its bound cohort ledger file",
    ):
        ae_stage.fit_side_local_ae_gmm_stage(
            side="long",
            cohort_ledger=cohort,
            cohort_ledger_path=cohort_path,
            authorized_population_ledger_path=population_path,
            feature_loader=lambda _rows, _columns: calls.append("loader"),
            input_features=["observable_a", "observable_b"],
            output_dir=tmp_path / "out",
            source_hashes=source_hashes,
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            min_reference_rows=3,
            resource_guard=_RecordingGuard(),
        )
    assert calls == []


def test_rejects_side_or_outcome_columns_before_feature_loader(
    tmp_path: Path,
) -> None:
    cohort = _cohort(include_after_reference=False)
    cohort_path, population_path, source_hashes = _write_ledgers(tmp_path, cohort)
    with pytest.raises(
        ae_stage.PackBSideLocalAEStageError,
        match="identity, side, or outcome-derived",
    ):
        ae_stage.fit_side_local_ae_gmm_stage(
            side="long",
            cohort_ledger=cohort,
            cohort_ledger_path=cohort_path,
            authorized_population_ledger_path=population_path,
            feature_loader=lambda _rows, _columns: pd.DataFrame(),
            input_features=["observable_a", "side_name"],
            output_dir=tmp_path / "out",
            source_hashes=source_hashes,
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            min_reference_rows=3,
            resource_guard=_RecordingGuard(),
        )


def test_disabled_fit_fails_closed_without_publishing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ae_stage.PackBSideLocalAEStageError, match="state is disabled"):
        _run(
            tmp_path,
            monkeypatch,
            fake_state={"enabled": False, "reason": "no_valid_gmm_config"},
        )
    assert not (tmp_path / "out" / "ae_gmm_state.pkl").exists()


def test_rejects_unbound_feature_loader_before_loading_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = _cohort(include_after_reference=False)
    cohort_path, population_path, source_hashes = _write_ledgers(tmp_path, cohort)
    calls: list[str] = []

    def unbound_loader(_rows: pd.DataFrame, _columns: list[str]) -> pd.DataFrame:
        calls.append("loader")
        return pd.DataFrame()

    monkeypatch.setattr(
        ae_stage,
        "fit_ae_gmm_state",
        lambda *_args, **_kwargs: calls.append("fit"),
    )
    with pytest.raises(
        ae_stage.PackBSideLocalAEStageError,
        match="frozen Pack-B static-loader evidence contract",
    ):
        ae_stage.fit_side_local_ae_gmm_stage(
            side="long",
            cohort_ledger=cohort,
            cohort_ledger_path=cohort_path,
            authorized_population_ledger_path=population_path,
            feature_loader=unbound_loader,
            input_features=["observable_a", "observable_b"],
            output_dir=tmp_path / "out",
            source_hashes=source_hashes,
            source_revision=SOURCE_REVISION,
            fixed_calendar_sha256=CALENDAR_SHA256,
            max_train_rows=3,
            gmm_max_train_rows=3,
            min_reference_rows=3,
            resource_guard=_RecordingGuard(),
        )
    assert calls == []
