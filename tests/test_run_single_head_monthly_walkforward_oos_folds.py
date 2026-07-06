from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_single_head_monthly_walkforward_oos as wf


def test_default_folds_remain_apr_may_jun_only() -> None:
    folds = wf._folds("exp")

    assert [fold.name for fold in folds] == [
        "train_through_march_score_april",
        "train_through_april_score_may",
        "train_through_may_score_june",
    ]
    assert [fold.run_id for fold in folds] == [
        "exp_train_march_score_april",
        "exp_train_april_score_may",
        "exp_train_may_score_june",
    ]


def test_include_july_fold_adds_train_through_june_score_july() -> None:
    folds = wf._folds("exp", include_july_fold=True)
    july = folds[-1]

    assert len(folds) == 4
    assert july.name == "train_through_june_score_july"
    assert july.run_id == "exp_train_june_score_july"
    assert july.train_end == pd.Timestamp("2026-06-30 11:00:00", tz="UTC")
    assert july.policy_start == pd.Timestamp("2026-07-01 00:00:00", tz="UTC")
    assert july.policy_split == pd.Timestamp("2026-07-16 00:00:00", tz="UTC")
    assert july.policy_end == pd.Timestamp("2026-08-01 00:00:00", tz="UTC")


def test_partial_july_fold_uses_bounded_midpoint_split() -> None:
    folds = wf._folds(
        "exp",
        include_july_fold=True,
        july_policy_end="2026-07-03 00:00:00",
    )
    july = folds[-1]

    assert july.policy_start == pd.Timestamp("2026-07-01 00:00:00", tz="UTC")
    assert july.policy_split == pd.Timestamp("2026-07-02 00:00:00", tz="UTC")
    assert july.policy_end == pd.Timestamp("2026-07-03 00:00:00", tz="UTC")


def test_july_fold_rejects_empty_policy_window() -> None:
    with pytest.raises(ValueError, match="policy_end must be after policy_start"):
        wf._folds(
            "exp",
            include_july_fold=True,
            july_policy_end="2026-07-01 00:00:00",
        )


def test_filter_folds_accepts_july_alias() -> None:
    folds = wf._folds("exp", include_july_fold=True)

    selected = wf._filter_folds(folds, ["july"])

    assert [fold.name for fold in selected] == ["train_through_june_score_july"]


def test_filter_folds_rejects_unknown_fold() -> None:
    folds = wf._folds("exp")

    with pytest.raises(ValueError, match="Unknown fold filter"):
        wf._filter_folds(folds, ["july"])


def _write_label_artifact(
    data_root: Path,
    *,
    run_id: str,
    strategy_id: str,
    timestamps: list[str],
) -> None:
    label_dir = data_root / "artifacts" / run_id / "labels"
    label_dir.mkdir(parents=True)
    label_key = wf._label_key(strategy_id)
    pd.DataFrame({"__ts__": pd.to_datetime(timestamps, utc=True)}).to_parquet(
        label_dir / f"{label_key}.parquet",
        index=False,
    )
    (label_dir / "labels_manifest.json").write_text(
        json.dumps({"datasets": {label_key: {"path": f"{label_key}.parquet"}}}),
        encoding="utf-8",
    )


def test_labels_ready_can_require_timestamp_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "DATA_ROOT", tmp_path / "data_perp")
    _write_label_artifact(
        wf.DATA_ROOT,
        run_id="labels",
        strategy_id="long_test",
        timestamps=["2026-06-30T06:00:00Z"],
    )

    assert wf._labels_ready("labels", "long_test")
    assert wf._labels_ready(
        "labels",
        "long_test",
        min_label_max_ts=pd.Timestamp("2026-06-30T06:00:00Z"),
    )
    assert not wf._labels_ready(
        "labels",
        "long_test",
        min_label_max_ts=pd.Timestamp("2026-07-01T00:00:00Z"),
    )


def _write_source_selection_artifacts(data_root: Path) -> str:
    run_id = "source"
    run_root = data_root / "artifacts" / run_id
    registry = run_root / "strategy_registry" / "deployed_four_heads_perps.csv"
    registry.parent.mkdir(parents=True)
    rows = [
        {"strategy_id": "long_a", "trade_side": "long", "side": "long"},
        {"strategy_id": "long_b", "trade_side": "long", "side": "long"},
        {"strategy_id": "short_a", "trade_side": "short", "side": "short"},
        {"strategy_id": "short_b", "trade_side": "short", "side": "short"},
    ]
    with registry.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    candidates = run_root / "simple_policy_optimiser" / "simple_policy_candidates_broad.parquet"
    candidates.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-06-02T00:00:00Z",
                "strategy_id": "long_a",
                "side": "long",
                "net_return": 0.10,
                "rank_pct": 0.8,
            },
            {
                "timestamp": "2026-06-03T00:00:00Z",
                "strategy_id": "long_b",
                "side": "long",
                "net_return": 0.20,
                "rank_pct": 0.9,
            },
            {
                "timestamp": "2026-06-02T00:00:00Z",
                "strategy_id": "short_a",
                "side": "short",
                "net_return": 0.05,
                "rank_pct": 0.7,
            },
            {
                "timestamp": "2026-06-03T00:00:00Z",
                "strategy_id": "short_b",
                "side": "short",
                "net_return": 0.30,
                "rank_pct": 0.95,
            },
            {
                "timestamp": "2026-05-31T23:00:00Z",
                "strategy_id": "long_a",
                "side": "long",
                "net_return": 10.0,
                "rank_pct": 1.0,
            },
        ]
    ).to_parquet(candidates, index=False)
    return run_id


def test_best_per_side_selection_picks_real_long_and_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "DATA_ROOT", tmp_path / "data_perp")
    source_run_id = _write_source_selection_artifacts(wf.DATA_ROOT)

    selection = wf._select_june_strategy_set(
        source_run_id,
        selection_mode="best_per_side",
        sides=["long", "short"],
    )

    assert selection["strategy_ids"] == ["long_b", "short_b"]
    assert [row["side"] for row in selection["selected_strategies"]] == ["long", "short"]
    assert selection["selection_mode"] == "best_per_side"


def test_overall_selection_keeps_legacy_single_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "DATA_ROOT", tmp_path / "data_perp")
    source_run_id = _write_source_selection_artifacts(wf.DATA_ROOT)

    selection = wf._select_june_strategy_set(
        source_run_id,
        selection_mode="overall",
    )

    assert selection["strategy_ids"] == ["short_b"]
    assert selection["strategy_id"] == "short_b"
    assert selection["selection_mode"] == "overall"


def test_multi_strategy_registry_and_commands_are_side_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "DATA_ROOT", tmp_path / "data_perp")
    rows = [
        {"strategy_id": "long_b", "trade_side": "long", "side": "long"},
        {"strategy_id": "short_b", "trade_side": "short", "side": "short"},
    ]

    path = wf._write_strategy_registry("run", rows)
    written = pd.read_csv(path)
    policy_cmd = wf._policy_oos_cmd("run", ["long_b", "short_b"])
    simple_cmd = wf._simple_policy_cmd("run", ["long_b", "short_b"])

    assert written["strategy_id"].tolist() == ["long_b", "short_b"]
    assert written["trade_side"].tolist() == ["long", "short"]
    assert policy_cmd.count("--strategy-id") == 2
    assert policy_cmd[policy_cmd.index("--strategy-id") + 1] == "long_b"
    assert policy_cmd[policy_cmd.index("--strategy-id", policy_cmd.index("--strategy-id") + 1) + 1] == "short_b"
    assert simple_cmd[-2:] == ["--strategy-ids", "long_b,short_b"]


def test_labels_env_can_enable_policy_net_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wf, "ROOT", tmp_path)

    env = wf._labels_env(
        label_run_id="labels_policy_net",
        source_run_id="source",
        feature_source_run_id="features",
        label_ablation_mode="",
        label_policy_net_replay=True,
        label_policy_net_replay_min_coverage=0.95,
        registry_path=tmp_path / "registry.csv",
        strategy_id="long_a,short_b",
    )

    assert env["EPM_LABEL_POLICY_NET_REPLAY_ENABLED"] == "1"
    assert env["EPM_LABEL_POLICY_NET_REPLAY_MIN_COVERAGE"] == "0.95"
    assert env["EPM_LABEL_STRATEGY_IDS"] == "long_a,short_b"
