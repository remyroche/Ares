from __future__ import annotations

import json
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd

from extreme_price_movements.inference import p8u_regular_state_forward as forwarder
from extreme_price_movements.inference.p8u_single_timestamp_graph import P8UOneTimestampExecutor
from extreme_price_movements.inference.p8u_stateful_single_timestamp_executor import (
    P8UStatefulSingleTimestampExecutor,
)
from extreme_price_movements.inference.p8u_staged_timestamp_executor import DIRECT_EXPENSIVE_FEATURES
from extreme_price_movements.inference.p8u_regular_state_forward import target_free_candidates


def test_target_free_candidates_are_complete_and_clocked() -> None:
    frame = target_free_candidates(
        ("AAA/USD:USD", "BBB/USD:USD"), timestamp="2026-08-14T12:00:00Z"
    )

    assert frame["candidate_id"].tolist() == [
        "AAA/USD:USD|long|2026-08-14T12:00:00Z",
        "BBB/USD:USD|long|2026-08-14T12:00:00Z",
    ]
    assert frame["__ts__"].nunique() == 1
    assert frame["__decision_ts__"].nunique() == 1
    assert frame["__decision_ts__"].iloc[0] == pd.Timestamp("2026-08-14T13:00:00Z")
    assert not frame.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any()


def test_target_free_candidates_normalises_naive_timestamp_to_utc() -> None:
    frame = target_free_candidates(("AAA/USD:USD",), timestamp="2026-08-14 12:00:00")
    assert frame["__ts__"].iloc[0] == pd.Timestamp("2026-08-14T12:00:00Z")


def test_stateful_executor_hides_later_source_rows_before_scoring() -> None:
    first = pd.Timestamp("2026-08-14T12:00:00Z")
    later = first + pd.Timedelta(hours=1)
    source = {
        "symbols": ("AAA/USD:USD",),
        "metadata": {"unchanged": True},
        "panel": {
            "close": pd.DataFrame(
                [[1.0], [2.0]],
                index=[first, later],
                columns=["AAA/USD:USD"],
            ),
        },
    }

    bounded = P8UStatefulSingleTimestampExecutor._source_as_of(source, timestamp=first)

    assert bounded["panel"]["close"].index.tolist() == [first]
    assert source["panel"]["close"].index.tolist() == [first, later]
    assert bounded["metadata"] == {"unchanged": True}


def test_persisted_state_contract_preserves_internal_helper_keys(tmp_path) -> None:
    root = tmp_path / "state"
    root.mkdir()
    symbols = ("AAA/USD:USD", "BBB/USD:USD")
    keys = ("model_field", "causal_helper_field")
    database = root / "nested_derived_state.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    contract_hash = forwarder._nested_contract_hash(
        columns=symbols,
        feature_keys=keys,
        max_rows=1536,
    )
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("schema", forwarder.NESTED_SCHEMA),
            ("contract_hash", contract_hash),
            ("columns_json", json.dumps(list(symbols))),
            ("feature_keys_json", json.dumps(list(keys))),
            ("max_rows", "1536"),
        ],
    )
    connection.commit()
    connection.close()

    resolved, source, key_hash = forwarder._persisted_state_contract_features(
        root,
        symbols=symbols,
        required_features=("model_field",),
    )

    assert resolved == tuple(sorted(keys))
    assert source == "persisted_nested_state_metadata"
    assert len(key_hash) == 64


def test_exact_full_bootstrap_is_accepted_only_with_declared_contract(tmp_path) -> None:
    root = tmp_path / "bootstrap"
    state = root / "state"
    state.mkdir(parents=True)
    (state / "state.bin").write_bytes(b"exact-full-state")
    (root / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "strict_r3_p8u_canonical_transform_state_bootstrap_v1",
                "status": "bootstrapped_unactivated",
                "initial_seed": "exact_full",
                "state_scope": "test-scope",
                "features": 175,
                "symbols": 160,
                "history_start": "2026-06-26T14:00:00+00:00",
                "end_exclusive": "2026-08-29T14:00:00+00:00",
            }
        )
    )

    resolved_state, stamp, mode, identity = forwarder._bootstrap_state(
        root, state_scope="test-scope"
    )

    assert resolved_state == state
    assert stamp == pd.Timestamp("2026-08-29T13:00:00Z")
    assert mode == "research_exact_full_bootstrap"
    assert len(identity) == 64


def test_regular_state_forward_advances_one_source_row_per_commit(tmp_path, monkeypatch) -> None:
    symbols = tuple(f"S{i:03d}" for i in range(160))
    index = pd.date_range("2026-08-14T12:00:00Z", periods=2, freq="h")
    panel = {
        field: pd.DataFrame(np.ones((2, 160), dtype=np.float32), index=index, columns=symbols)
        for field in P8UOneTimestampExecutor.SOURCE_FIELDS
    }
    source = {"symbols": symbols, "panel": panel}
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "state.bin").write_bytes(b"state")
    (bootstrap / "canonical_state_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "strict_r3_p8u_canonical_state_checkpoint_v1",
                "state_scope": "test-scope",
                "as_of_timestamp": "2026-08-14T12:00:00+00:00",
            }
        )
    )
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")

    class FakeStage:
        def __init__(self, **_kwargs):
            pass

        def materialize_regular_feature_state_snapshot(self, *, candidates, panel, **_kwargs):
            assert all(len(value) == 1 for value in panel.values() if isinstance(value, pd.DataFrame))
            return candidates.assign(regular_field=np.float32(1.0))

    fake_stack = SimpleNamespace(
        preproduction=SimpleNamespace(
            feature_plan=lambda: SimpleNamespace(
                router_features=("regular_field",),
                base_features=(),
                under_features=(),
                full_union=(*DIRECT_EXPENSIVE_FEATURES, "regular_field"),
            )
        )
    )
    monkeypatch.setattr(forwarder, "P8URouterFirstVectorizedStage", FakeStage)
    monkeypatch.setattr(
        forwarder,
        "_advance_perp_tail_supplement",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported supplement invoked")),
    )

    def copy_tree(source_path, destination_path):
        shutil.copytree(source_path, destination_path)
        return "copy"

    output = tmp_path / "out"
    result = forwarder.forward_regular_state(
        bundle=bundle,
        bootstrap_state_root=bootstrap,
        output_root=output,
        source=source,
        state_scope="test-scope",
        state_components=("raw",),
        through="2026-08-14T13:00:00Z",
        stack_loader=lambda *_args, **_kwargs: fake_stack,
        tree_cloner=copy_tree,
    )

    assert result.committed_timestamps == (pd.Timestamp("2026-08-14T13:00:00Z"),)
    receipt = json.loads((output / "commits" / "20260814T130000Z" / "receipt.json").read_text())
    assert receipt["source_rows_fed"] == 1
    assert receipt["broad_retained_tail_feature_graph_called"] is False
    assert receipt["supplemental_regular_fields"] == []
    ledger = json.loads((output / forwarder.LEDGER).read_text())
    assert ledger["last_source_timestamp"] == "2026-08-14T13:00:00+00:00"
    assert len(ledger["active_state_chain_identity"]) == 64
    assert ledger["state_contract_feature_count"] == len(DIRECT_EXPENSIVE_FEATURES) + 1
