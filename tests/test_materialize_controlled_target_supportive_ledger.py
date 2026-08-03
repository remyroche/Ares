from __future__ import annotations

import pandas as pd

from scripts.materialize_controlled_target_supportive_ledger import fold_table, protocol_folds, resolve_target_pack


def test_protocol_is_the_fixed_12_plus_4_plus_4_chronological_split() -> None:
    timestamps = pd.to_datetime(["2023-04-01", "2024-03-31 23:00", "2024-04-01", "2024-07-31 23:00", "2024-08-01", "2024-11-30 23:00"], utc=True, format="mixed")
    folds = protocol_folds(timestamps)
    assert folds.oof_fold.tolist() == ["base_train", "base_train", "meta_train", "meta_train", "meta_oos", "meta_oos"]
    assert folds.fold_order.tolist() == [0, 0, 1, 1, 2, 2]
    table = fold_table()
    assert table.start_utc.tolist()[1] == pd.Timestamp("2024-04-01T00:00:00Z")
    assert table.end_exclusive_utc.tolist()[-1] == pd.Timestamp("2024-12-01T00:00:00Z")


def test_target_resolution_prefers_completed_v2_and_never_staging(tmp_path, monkeypatch) -> None:
    import scripts.materialize_controlled_target_supportive_ledger as runner
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    for root in (v1, v2):
        root.mkdir()
        for name in ("primary_labels.parquet", "supportive_labels.parquet", "manifest.json", "execution_target_contract.json"):
            (root / name).write_text("x")
    monkeypatch.setattr(runner, "DEFAULT_TARGET_V1", v1)
    monkeypatch.setattr(runner, "DEFAULT_TARGET_V2", v2)
    selected, reason = resolve_target_pack()
    assert selected == v2
    assert reason == "v2_preferred"
