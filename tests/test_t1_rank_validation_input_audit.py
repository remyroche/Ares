import json
from pathlib import Path

import pandas as pd

from scripts.audit_t1_rank_validation_inputs import audit_candidate_root, discover_candidate_roots, _render_report


REQUIRED_VALUES = {
    "timestamp": pd.date_range("2026-06-23", periods=3, freq="h", tz="UTC"),
    "symbol": ["BTC", "ETH", "SOL"],
    "side": ["short", "short", "short"],
    "strategy_id": ["short_boll_s1", "short_boll_s1", "short_asset_s1"],
    "head": ["short_boll", "short_boll", "short_asset"],
    "calibrated_score": [0.71, 0.73, 0.76],
    "normalized_rank_score": [0.80, 0.85, 0.90],
    "strategy_rank_pct": [0.80, 0.85, 0.90],
    "policy_rank_pct": [0.80, 0.85, 0.90],
    "base_strategy_threshold": [0.70, 0.70, 0.70],
    "entry_price": [100.0, 100.0, 100.0],
    "exit_price": [101.0, 99.0, 102.0],
    "exit_timestamp": pd.date_range("2026-06-23 04:00", periods=3, freq="h", tz="UTC"),
    "net_return": [0.01, -0.01, 0.02],
    "gross_return": [0.011, -0.009, 0.021],
    "holding_bars": [4, 4, 4],
    "simple_policy_exit_reason": ["tp", "sl", "tp"],
}


def _write_candidate_root(
    tmp_path: Path,
    *,
    name: str,
    timestamps: pd.DatetimeIndex | None = None,
    native_manifest: dict | None = None,
    t1_manifest: dict | None = None,
    t1_anchor_manifest: dict | None = None,
    write_deployable: bool = True,
) -> Path:
    root = tmp_path / name
    policy_dir = root / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True)
    frame = pd.DataFrame(REQUIRED_VALUES)
    if timestamps is not None:
        frame["timestamp"] = timestamps
    frame.to_parquet(policy_dir / "simple_policy_candidates_broad.parquet", index=False)
    if write_deployable:
        frame.to_parquet(policy_dir / "simple_policy_candidates.parquet", index=False)
    if native_manifest is not None:
        (root / "live_ledger_native_materialization_manifest.json").write_text(
            json.dumps(native_manifest),
            encoding="utf-8",
        )
    if t1_manifest is not None:
        (root / "t1_repaired_static_baseline_manifest.json").write_text(
            json.dumps(t1_manifest),
            encoding="utf-8",
        )
    if t1_anchor_manifest is not None:
        (root / "t1_anchor_scored_candidate_manifest.json").write_text(
            json.dumps(t1_anchor_manifest),
            encoding="utf-8",
        )
    return root


def test_audit_accepts_later_anchor_score_ledger_with_existing_score_source(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame({"score": [0.1]}).to_parquet(score_path, index=False)
    root = _write_candidate_root(
        tmp_path,
        name="anchor_later",
        native_manifest={
            "score_diagnostics": {"score_column": "reliability_anchor_only_score"},
            "score_path": str(score_path),
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is True
    assert row["score_is_anchor_compatible"] is True
    assert row["score_path_exists"] is True
    assert row["period_after_min_timestamp"] is True
    assert row["rejection_reasons"] == ""


def test_audit_rejects_later_native_reliability_blend_ledger(tmp_path: Path) -> None:
    root = _write_candidate_root(
        tmp_path,
        name="native_blend_later",
        native_manifest={
            "score_diagnostics": {"score_column": "reliability_blend_score"},
            "score_path": str(tmp_path / "missing_scores.parquet"),
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is False
    assert row["period_after_min_timestamp"] is True
    assert row["score_is_anchor_compatible"] is False
    assert row["score_path_exists"] is False
    assert "referenced_score_path_missing" in row["rejection_reasons"]
    assert "score_not_anchor_compatible" in row["rejection_reasons"]


def test_audit_rejects_native_manifest_generic_calibrated_score(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame({"calibrated_score": [0.1]}).to_parquet(score_path, index=False)
    root = _write_candidate_root(
        tmp_path,
        name="native_generic_calibrated",
        native_manifest={
            "score_diagnostics": {"score_column": "calibrated_score"},
            "score_path": str(score_path),
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is False
    assert row["score_path_exists"] is True
    assert row["score_is_anchor_compatible"] is False
    assert "generic_calibrated_score_requires_t1_manifest" in row["rejection_reasons"]
    assert "score_not_anchor_compatible" in row["rejection_reasons"]


def test_audit_rejects_pre_boundary_anchor_ledger(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame({"score": [0.1]}).to_parquet(score_path, index=False)
    root = _write_candidate_root(
        tmp_path,
        name="anchor_pre_boundary",
        timestamps=pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC"),
        native_manifest={
            "score_diagnostics": {"score_column": "reliability_anchor_only_score"},
            "score_path": str(score_path),
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is False
    assert row["score_is_anchor_compatible"] is True
    assert row["period_after_min_timestamp"] is False
    assert row["rejection_reasons"] == "period_not_after_min_timestamp"


def test_audit_accepts_t1_manifest_anchor_meta_score_without_native_score_path(tmp_path: Path) -> None:
    root = _write_candidate_root(
        tmp_path,
        name="t1_manifest_later",
        t1_manifest={
            "active_stack": {
                "active_score_column": "calibrated_score",
                "score_path": "anchor_meta_calibrated_score",
            }
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is True
    assert row["score_column"] == "calibrated_score"
    assert row["score_is_anchor_compatible"] is True
    assert row["score_path_exists"] is None
    assert row["rejection_reasons"] == ""


def test_audit_accepts_t1_anchor_scored_candidate_manifest(tmp_path: Path) -> None:
    score_path = tmp_path / "anchor_scores.parquet"
    pd.DataFrame({"calibrated_score": [0.1]}).to_parquet(score_path, index=False)
    root = _write_candidate_root(
        tmp_path,
        name="t1_anchor_scored_later",
        t1_anchor_manifest={
            "score_ledger_path": str(score_path),
            "score_contract": {
                "score_column": "calibrated_score",
                "score_source": "live_finalfit_anchor_meta_score",
                "native_reliability_blend_active": False,
                "qfail_active": False,
                "market_state_threshold_controller_active": False,
            },
        },
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is True
    assert row["score_column"] == "calibrated_score"
    assert row["score_source"] == "live_finalfit_anchor_meta_score"
    assert row["score_path_exists"] is True
    assert row["score_is_anchor_compatible"] is True
    assert row["rejection_reasons"] == ""


def test_audit_rejects_missing_deployable_candidates(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame({"score": [0.1]}).to_parquet(score_path, index=False)
    root = _write_candidate_root(
        tmp_path,
        name="anchor_missing_deployable",
        native_manifest={
            "score_diagnostics": {"score_column": "reliability_anchor_only_score"},
            "score_path": str(score_path),
        },
        write_deployable=False,
    )

    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-22T23:59:59Z"))

    assert row["eligible_for_t1_rank_validation"] is False
    assert row["broad_exists"] is True
    assert row["deployable_exists"] is False
    assert row["rejection_reasons"] == "missing_deployable_candidates"


def test_discover_candidate_roots_finds_policy_candidate_ledgers(tmp_path: Path) -> None:
    root_a = _write_candidate_root(tmp_path, name="a")
    root_b = _write_candidate_root(tmp_path, name="nested/b")
    noise = tmp_path / "noise" / "simple_policy_optimiser"
    noise.mkdir(parents=True)
    (noise / "not_candidates.parquet").write_text("x", encoding="utf-8")

    roots = discover_candidate_roots([tmp_path])

    assert roots == sorted([root_a, root_b], key=lambda p: str(p))


def test_discover_candidate_roots_accepts_direct_broad_candidate_file(tmp_path: Path) -> None:
    root = _write_candidate_root(tmp_path, name="direct")
    broad = root / "simple_policy_optimiser" / "simple_policy_candidates_broad.parquet"

    roots = discover_candidate_roots([broad])

    assert roots == [root]


def test_render_report_includes_compact_summary(tmp_path: Path) -> None:
    root = _write_candidate_root(
        tmp_path,
        name="old",
        timestamps=pd.date_range("2026-06-20", periods=3, freq="h", tz="UTC"),
        t1_manifest={
            "active_stack": {
                "active_score_column": "calibrated_score",
                "score_path": "anchor_meta_calibrated_score",
            }
        },
    )
    row = audit_candidate_root(root, min_timestamp=pd.Timestamp("2026-06-23T09:00:00Z"))
    report = _render_report(tmp_path / "out", pd.DataFrame([row]))

    assert "## Summary" in report
    assert "Audited roots: `1`" in report
    assert "Eligible roots: `0`" in report
    assert "| period_not_after_min_timestamp | 1 |" in report
