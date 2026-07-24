from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_packb_pre_march_side_ae as runner
from scripts.audit_full_pipeline_migration import hash_path


def test_coverage_segments_are_disjoint_and_cover_the_reference() -> None:
    ledger = pd.DataFrame(
        {
            "candidate_id": ["begin", "middle", "end"],
            "__ts__": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-04-01T00:00:00Z",
                    "2025-08-01T00:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["A", "B", "C"],
        }
    )
    segments = runner._coverage_segments(ledger)
    assert set(segments) == {"beginning", "middle", "end"}
    assert {
        name: frame["candidate_id"].tolist() for name, frame in segments.items()
    } == {
        "beginning": ["begin"],
        "middle": ["middle"],
        "end": ["end"],
    }


def test_feature_store_revalidation_matches_content_tree_and_rejects_change(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "one.txt").write_text("one\n", encoding="utf-8")
    digest = hash_path(store)
    inventory = {
        "inventory": {
            "items": [
                {
                    "id": "canonical_feature_store",
                    "sha256": digest["sha256"],
                    "bytes": digest["bytes"],
                    "files": digest["files"],
                    "directories": digest["directories"],
                }
            ]
        }
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    binding = runner._feature_inventory_binding(inventory_path)
    assert (
        runner._revalidate_feature_store(store, binding)["sha256"] == digest["sha256"]
    )

    (store / "one.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        runner.PackBSideAERunnerError, match="changed since the R0 inventory"
    ):
        runner._revalidate_feature_store(store, binding)
