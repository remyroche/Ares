from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements import (
    packb_pre_march_source_authorization as authorization,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_rows(
    side: str, *, candidate_id: str, signal: str, decision: str | None = None
) -> pd.DataFrame:
    signal_ts = pd.Timestamp(signal)
    decision_ts = (
        pd.Timestamp(decision)
        if decision is not None
        else signal_ts + pd.Timedelta(hours=1)
    )
    return pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            "__ts__": [signal_ts],
            "__decision_ts__": [decision_ts],
            "side_name": [side],
            # The authorizer must not need model features or targets.
            "unrelated_feature": [1.0],
        }
    )


def _write_label_source(
    tmp_path: Path, frames: dict[str, pd.DataFrame]
) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    labels.mkdir()
    for name, frame in frames.items():
        frame.to_parquet(labels / name, index=False)
    audit = labels / "causal_path_invariant_audit.json"
    audit.write_text(
        json.dumps(
            {
                "files": len(frames),
                "per_file": [{"file": name} for name in sorted(frames)],
            }
        ),
        encoding="utf-8",
    )
    return labels, audit


def _write_side_source(
    tmp_path: Path, side: str
) -> authorization.SideSourceAuthorization:
    artifact_dir = tmp_path / "artifacts" / side
    artifact_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for name in ("state", "features", "params"):
        path = artifact_dir / f"{name}.bin"
        path.write_text(f"{side}-{name}", encoding="utf-8")
        paths[name] = path
    return authorization.SideSourceAuthorization(
        ae_gmm_state_path=paths["state"],
        ae_gmm_state_sha256=_digest(paths["state"]),
        ae_gmm_state_scope=side,
        ae_gmm_reference_label_resolution_max_utc="2026-02-28T23:00:00Z",
        feature_contract_path=paths["features"],
        feature_contract_sha256=_digest(paths["features"]),
        feature_contract_scope=side,
        feature_selection_label_resolution_max_utc="2026-02-28T23:00:00Z",
        parameter_path=paths["params"],
        parameter_sha256=_digest(paths["params"]),
        parameter_scope=side,
        hpo_label_resolution_max_utc="2026-02-28T23:00:00Z",
    )


def _valid_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, authorization.SideSourceAuthorization]]:
    labels, audit = _write_label_source(
        tmp_path,
        {
            "long.parquet": pd.concat(
                [
                    _label_rows(
                        "long", candidate_id="long-pre", signal="2026-02-27T22:00:00Z"
                    ),
                    _label_rows(
                        "long", candidate_id="long-later", signal="2026-03-04T00:00:00Z"
                    ),
                ],
                ignore_index=True,
            ),
            "short.parquet": pd.concat(
                [
                    _label_rows(
                        "short", candidate_id="short-pre", signal="2026-02-27T22:00:00Z"
                    ),
                    _label_rows(
                        "short",
                        candidate_id="short-later",
                        signal="2026-03-04T00:00:00Z",
                    ),
                ],
                ignore_index=True,
            ),
        },
    )
    return (
        labels,
        audit,
        {
            side: _write_side_source(tmp_path, side)
            for side in authorization.CANONICAL_SIDES
        },
    )


def test_authorizes_exact_inventory_and_reports_only_actual_pre_march_rows(
    tmp_path: Path,
) -> None:
    labels, audit, sources = _valid_inputs(tmp_path)

    report = authorization.authorize_pre_march_packb_sources(
        labels_dir=labels,
        causal_audit_path=audit,
        side_sources=sources,
        batch_rows=1,
    )

    assert report["schema"] == authorization.AUTHORIZATION_SCHEMA
    assert report["label_inventory"]["canonical_shard_count"] == 2
    assert report["label_rows_scanned"] == 4
    assert report["streaming_contract"]["feature_or_target_columns_loaded"] is False
    for side in authorization.CANONICAL_SIDES:
        population = report["authorized_population_by_side"][side]
        assert population["authorized_rows"] == 1
        assert population["excluded_rows_at_or_after_cutoff"] == 1
        assert (
            population["authorized_label_resolution_max_utc"]
            == "2026-02-28T23:00:00+00:00"
        )
        assert len(population["authorized_candidate_stream_sha256"]) == 64
        artifacts = report["side_source_artifacts"][side]["artifacts"]
        assert {name: artifact["scope"] for name, artifact in artifacts.items()} == {
            "ae_gmm_state": side,
            "feature_contract": side,
            "parameter": side,
        }


def test_pre_fit_population_and_post_fit_artifact_checks_are_separate(
    tmp_path: Path,
) -> None:
    labels, audit, sources = _valid_inputs(tmp_path)

    population = authorization.preflight_pre_march_packb_population(
        labels_dir=labels,
        causal_audit_path=audit,
        batch_rows=1,
    )
    artifacts = authorization.verify_pre_march_side_artifacts(
        side_sources=sources,
    )

    assert population["status"] == "AUTHORIZED_PRE_MARCH_POPULATION"
    assert set(population["authorized_population_by_side"]) == {"long", "short"}
    assert set(artifacts) == {"long", "short"}


@pytest.mark.parametrize("kind", ["extra", "missing"])
def test_rejects_non_exact_causal_audit_inventory(tmp_path: Path, kind: str) -> None:
    labels, audit, sources = _valid_inputs(tmp_path)
    if kind == "extra":
        _label_rows(
            "short", candidate_id="stale", signal="2026-02-27T22:00:00Z"
        ).to_parquet(labels / "stale_short_7.parquet", index=False)
        expected = "unlisted parquet shards"
    else:
        (labels / "short.parquet").unlink()
        expected = "missing canonical shards"

    with pytest.raises(authorization.PackBSourceAuthorizationError, match=expected):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels, causal_audit_path=audit, side_sources=sources
        )


def test_rejects_duplicate_candidate_ids_across_canonical_shards_in_batches(
    tmp_path: Path,
) -> None:
    labels, audit = _write_label_source(
        tmp_path,
        {
            "long.parquet": _label_rows(
                "long", candidate_id="same-id", signal="2026-02-27T22:00:00Z"
            ),
            "short.parquet": _label_rows(
                "short", candidate_id="same-id", signal="2026-02-27T22:00:00Z"
            ),
        },
    )
    sources = {
        side: _write_side_source(tmp_path, side)
        for side in authorization.CANONICAL_SIDES
    }

    with pytest.raises(
        authorization.PackBSourceAuthorizationError, match="duplicate candidate_id"
    ):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels,
            causal_audit_path=audit,
            side_sources=sources,
            batch_rows=1,
        )


def test_rejects_non_actual_decision_timestamp_before_it_can_authorize_labels(
    tmp_path: Path,
) -> None:
    labels, audit, sources = _valid_inputs(tmp_path)
    _label_rows(
        "short",
        candidate_id="bad-decision",
        signal="2026-02-27T22:00:00Z",
        decision="2026-02-27T22:30:00Z",
    ).to_parquet(labels / "short.parquet", index=False)

    with pytest.raises(
        authorization.PackBSourceAuthorizationError, match="decision_timestamp"
    ):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels, causal_audit_path=audit, side_sources=sources
        )


def test_rejects_post_cutoff_or_pooled_side_artifacts(tmp_path: Path) -> None:
    labels, audit, sources = _valid_inputs(tmp_path)
    with pytest.raises(
        authorization.PackBSourceAuthorizationError,
        match="uses a label resolved at/after",
    ):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels,
            causal_audit_path=audit,
            side_sources={
                **sources,
                "short": replace(
                    sources["short"],
                    hpo_label_resolution_max_utc="2026-03-01T00:00:00Z",
                ),
            },
        )

    long_state = sources["long"].ae_gmm_state_path
    pooled_short = replace(
        sources["short"],
        ae_gmm_state_path=long_state,
        ae_gmm_state_sha256=_digest(long_state),
    )
    with pytest.raises(
        authorization.PackBSourceAuthorizationError, match="distinct ae_gmm_state"
    ):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels,
            causal_audit_path=audit,
            side_sources={**sources, "short": pooled_short},
        )

    invalid_scope = replace(sources["short"], parameter_scope="global")
    with pytest.raises(
        authorization.PackBSourceAuthorizationError, match="parameter scope"
    ):
        authorization.authorize_pre_march_packb_sources(
            labels_dir=labels,
            causal_audit_path=audit,
            side_sources={**sources, "short": invalid_scope},
        )
