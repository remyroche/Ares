from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements import packb_side_stage_manifest as manifest_module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


SOURCE_HASHES = {
    "dec09_decisions_sha256": _sha("dec09-decisions"),
    "canonical_shard_inventory_sha256": _sha("canonical-shards"),
    "causal_audit_sha256": _sha("causal-audit"),
    "population_preflight_sha256": _sha("preflight"),
    "authorized_population_ledger_sha256": "0" * 64,
    "feature_store_inventory_sha256": _sha("feature-store"),
    "feature_store_inventory_evidence_sha256": _sha("feature-store-inventory-evidence"),
}
CALENDAR_SHA256 = _sha("fixed-inner-and-outer-calendar")
SOURCE_REVISION = "1" * 40


def _candidate_row(side: str, stage: str, signal: str) -> dict[str, object]:
    signal_ts = pd.Timestamp(signal)
    decision = signal_ts + pd.Timedelta(hours=1)
    return {
        "candidate_id": f"{side}-{stage}-{signal}",
        "side_name": side,
        "__ts__": signal_ts,
        "__decision_ts__": decision,
        "__label_resolution_ts__": decision + pd.Timedelta(hours=24),
    }


def _population_ledger(root: Path) -> Path:
    path = root / "authorized_population.parquet"
    if not path.exists():
        rows = [
            _candidate_row(side, stage, "2026-02-27T22:00:00Z")
            for side in manifest_module.CANONICAL_SIDES
            for stage in manifest_module.CANONICAL_STAGES
        ]
        pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _source_hashes(root: Path) -> dict[str, str]:
    ledger = _population_ledger(root)
    return {
        **SOURCE_HASHES,
        "authorized_population_ledger_sha256": manifest_module.sha256_file(ledger),
    }


def _candidate_stream(
    root: Path,
    *,
    side: str,
    stage: str,
    max_signal: str = "2026-02-27T22:00:00Z",
) -> dict[str, object]:
    path = root / "candidate_streams" / side / f"{stage}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _candidate_row(side, stage, max_signal)
    pd.DataFrame([row]).to_parquet(path, index=False)
    return {
        "path": str(path),
        "count": 1,
        "sha256": manifest_module.sha256_file(path),
        "signal_min_utc": pd.Timestamp(row["__ts__"]).isoformat(),
        "signal_max_utc": pd.Timestamp(row["__ts__"]).isoformat(),
        "decision_min_utc": pd.Timestamp(row["__decision_ts__"]).isoformat(),
        "decision_max_utc": pd.Timestamp(row["__decision_ts__"]).isoformat(),
        "label_resolution_min_utc": pd.Timestamp(
            row["__label_resolution_ts__"]
        ).isoformat(),
        "label_resolution_max_utc": pd.Timestamp(
            row["__label_resolution_ts__"]
        ).isoformat(),
    }


def _write_stage_manifest(
    root: Path,
    *,
    side: str,
    stage: str,
    artifact_text: str | None = None,
    artifact_scope: str | None = None,
    source_hashes: dict[str, str] | None = None,
    calendar_sha256: str = CALENDAR_SHA256,
    candidate_stream: dict[str, object] | None = None,
) -> Path:
    stage_dir = root / "manifests" / side
    artifact_dir = root / "artifacts" / side
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    kind = manifest_module._ARTIFACT_KIND_BY_STAGE[stage]
    artifact = artifact_dir / f"{stage}.bin"
    artifact.write_text(artifact_text or f"{side}-{stage}", encoding="utf-8")
    stage_config = artifact_dir / f"{stage}_config.json"
    stage_config.write_text(
        json.dumps({"side": side, "stage": stage}),
        encoding="utf-8",
    )
    population = _population_ledger(root)
    path = stage_dir / f"{stage}.json"
    payload = {
        "schema": manifest_module.SIDE_STAGE_MANIFEST_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "side": side,
        "stage": stage,
        "resolution_cutoff_utc": "2026-03-01T00:00:00Z",
        "actual_label_resolution_contract": (
            manifest_module.ACTUAL_LABEL_RESOLUTION_CONTRACT
        ),
        "source_hashes": source_hashes or _source_hashes(root),
        "authorized_population_ledger": {
            "path": str(Path("../..") / population.relative_to(root)),
            "sha256": manifest_module.sha256_file(population),
        },
        "candidate_stream": candidate_stream
        or _candidate_stream(root, side=side, stage=stage),
        "fixed_calendar_sha256": calendar_sha256,
        "stage_config": {
            "path": str(Path("../..") / stage_config.relative_to(root)),
            "sha256": manifest_module.sha256_file(stage_config),
        },
        "artifact": {
            "kind": kind,
            "path": str(Path("../..") / artifact.relative_to(root)),
            "sha256": manifest_module.sha256_file(artifact),
            "scope": artifact_scope or side,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _valid_bundle(root: Path) -> dict[str, dict[str, Path]]:
    return {
        side: {
            stage: _write_stage_manifest(root, side=side, stage=stage)
            for stage in manifest_module.CANONICAL_STAGES
        }
        for side in manifest_module.CANONICAL_SIDES
    }


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_validates_complete_side_local_bundle_against_actual_artifacts(
    tmp_path: Path,
) -> None:
    paths = _valid_bundle(tmp_path)

    report = manifest_module.validate_side_stage_manifest_bundle(
        paths,
        expected_source_revision=SOURCE_REVISION,
        expected_source_hashes=_source_hashes(tmp_path),
        expected_fixed_calendar_sha256=CALENDAR_SHA256,
    )

    assert report["status"] == "VALIDATED_POST_FIT_PRE_MARCH_SIDE_STAGES"
    assert report["source_hashes"] == _source_hashes(tmp_path)
    assert report["fixed_calendar_sha256"] == CALENDAR_SHA256
    for side in manifest_module.CANONICAL_SIDES:
        for stage in manifest_module.CANONICAL_STAGES:
            evidence = report["by_side"][side][stage]
            assert evidence["side"] == side
            assert evidence["stage"] == stage
            assert evidence["artifact"]["scope"] == side
            assert evidence["manifest_sha256"] == manifest_module.sha256_file(
                paths[side][stage]
            )


def test_rejects_candidate_timing_at_cutoff_from_manifest_evidence(
    tmp_path: Path,
) -> None:
    path = _write_stage_manifest(
        tmp_path,
        side="long",
        stage="hpo",
        candidate_stream=_candidate_stream(
            tmp_path,
            side="long",
            stage="hpo",
            max_signal="2026-02-27T23:00:00Z",
        ),
    )

    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="at/after the pre-March cutoff",
    ):
        manifest_module.validate_side_stage_manifest(path)


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("side_name", "short", "outside side"),
        ("candidate_id", "foreign-id", "absent from the authorized population"),
    ],
)
def test_rejects_wrong_side_or_foreign_candidate_evidence(
    tmp_path: Path,
    column: str,
    value: str,
    error: str,
) -> None:
    path = _write_stage_manifest(tmp_path, side="long", stage="feature_selection")
    payload = _payload(path)
    candidate_path = Path(payload["candidate_stream"]["path"])
    frame = pd.read_parquet(candidate_path)
    frame.loc[0, column] = value
    frame.to_parquet(candidate_path, index=False)
    payload["candidate_stream"]["sha256"] = manifest_module.sha256_file(candidate_path)
    _replace_payload(path, payload)

    with pytest.raises(manifest_module.PackBSideStageManifestError, match=error):
        manifest_module.validate_side_stage_manifest(path)


def test_rejects_artifact_changed_after_manifest_was_written(tmp_path: Path) -> None:
    path = _write_stage_manifest(tmp_path, side="long", stage="ae_gmm")
    payload = _payload(path)
    artifact_path = (path.parent / payload["artifact"]["path"]).resolve()
    artifact_path.write_text("modified-after-fit", encoding="utf-8")

    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="artifact SHA-256 does not match",
    ):
        manifest_module.validate_side_stage_manifest(path)


def test_rejects_pooled_learned_artifact_even_when_manifest_scopes_claim_sides(
    tmp_path: Path,
) -> None:
    paths = _valid_bundle(tmp_path)
    long_path = paths["long"]["ae_gmm"]
    short_path = paths["short"]["ae_gmm"]
    long_payload = _payload(long_path)
    short_payload = _payload(short_path)
    short_payload["artifact"]["path"] = long_payload["artifact"]["path"]
    short_payload["artifact"]["sha256"] = long_payload["artifact"]["sha256"]
    _replace_payload(short_path, short_payload)

    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="distinct learned ae_gmm artifact",
    ):
        manifest_module.validate_side_stage_manifest_bundle(paths)


def test_rejects_cross_side_candidate_evidence_file(tmp_path: Path) -> None:
    paths = _valid_bundle(tmp_path)
    long_path = paths["long"]["hpo"]
    short_path = paths["short"]["hpo"]
    long_payload = _payload(long_path)
    short_payload = _payload(short_path)
    short_payload["candidate_stream"]["path"] = long_payload["candidate_stream"]["path"]
    short_payload["candidate_stream"]["sha256"] = long_payload["candidate_stream"][
        "sha256"
    ]
    _replace_payload(short_path, short_payload)

    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="outside side",
    ):
        manifest_module.validate_side_stage_manifest_bundle(paths)


def test_rejects_wrong_source_or_calendar_binding(tmp_path: Path) -> None:
    path = _write_stage_manifest(tmp_path, side="short", stage="feature_selection")

    wrong_sources = _source_hashes(tmp_path)
    wrong_sources["population_preflight_sha256"] = _sha("other-preflight")
    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="source hashes do not match",
    ):
        manifest_module.validate_side_stage_manifest(
            path,
            expected_source_hashes=wrong_sources,
        )
    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="fixed calendar hash does not match",
    ):
        manifest_module.validate_side_stage_manifest(
            path,
            expected_fixed_calendar_sha256=_sha("wrong-calendar"),
        )


def test_immutable_writer_refuses_overwrite_and_reads_written_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_text("long-hpo", encoding="utf-8")
    path = tmp_path / "manifest.json"
    stage_config = tmp_path / "stage_config.json"
    stage_config.write_text("{}", encoding="utf-8")
    population = _population_ledger(tmp_path)
    candidate_stream = _candidate_stream(
        tmp_path,
        side="long",
        stage="hpo",
    )
    payload = {
        "schema": manifest_module.SIDE_STAGE_MANIFEST_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "side": "long",
        "stage": "hpo",
        "resolution_cutoff_utc": "2026-03-01T00:00:00Z",
        "actual_label_resolution_contract": (
            manifest_module.ACTUAL_LABEL_RESOLUTION_CONTRACT
        ),
        "source_hashes": _source_hashes(tmp_path),
        "authorized_population_ledger": {
            "path": "authorized_population.parquet",
            "sha256": manifest_module.sha256_file(population),
        },
        "candidate_stream": candidate_stream,
        "fixed_calendar_sha256": CALENDAR_SHA256,
        "stage_config": {
            "path": "stage_config.json",
            "sha256": manifest_module.sha256_file(stage_config),
        },
        "artifact": {
            "kind": "parameter",
            "path": "artifact.bin",
            "sha256": manifest_module.sha256_file(artifact),
            "scope": "long",
        },
    }

    digest = manifest_module.write_immutable_side_stage_manifest(path, payload)

    assert digest == manifest_module.sha256_file(path)
    assert manifest_module.validate_side_stage_manifest(path)["stage"] == "hpo"
    with pytest.raises(
        manifest_module.PackBSideStageManifestError,
        match="refusing to overwrite",
    ):
        manifest_module.write_immutable_side_stage_manifest(path, payload)
