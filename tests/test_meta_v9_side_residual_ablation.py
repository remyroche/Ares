import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.run_meta_v9_ev_mapped_side_residual_ablation as ablation
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS


def _selection_hpo_contract(tmp_path: Path, *, candidate_context: list[str] | None = None):
    handoff = tmp_path / "handoff.parquet"
    scored_ledger = tmp_path / "scored_ledger.parquet"
    global_predictions = tmp_path / "global_predictions.parquet"
    backbone_contract = tmp_path / "backbone.json"
    feature_dir = tmp_path / "features" / "20260301_000000"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for path, contents in (
        (handoff, b"handoff-v1"),
        (scored_ledger, b"ledger-v1"),
        (global_predictions, b"global-v1"),
        (backbone_contract, b'{"selected_feature_union":["context_a"]}'),
        (feature_dir / "symbol=BTC.parquet", b"feature-store-v1"),
    ):
        path.write_bytes(contents)
    return ablation._selection_hpo_contract(
        source_mode="current_handoff",
        handoff=handoff,
        scored_ledger=scored_ledger,
        global_predictions=global_predictions,
        feature_dir=feature_dir,
        feature_store_schema={"context_a", "context_b"},
        backbone_contract=backbone_contract,
        candidate_context=candidate_context or ["context_a", "context_b"],
        backbone_score="base",
        backbone_score_col="score_base",
        calibration_start=pd.Timestamp("2026-03-01", tz="UTC"),
        calibration_end=pd.Timestamp("2026-04-01", tz="UTC"),
        eval_start="2026-04-01",
        eval_end="2026-07-11",
        selection_mode="staged_mda",
        context_contract_only=False,
        excluded_context_prefixes=(),
        selection_max_rows=45_000,
        hpo_max_rows=45_000,
        hpo_trials=150,
        hpo_patience=40,
        seed=20260713,
        fixed_hpo_params_manifest=None,
    )


def _write_completed_selection_hpo_manifest(directory: Path, contract: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ablation.SELECTION_HPO_MANIFEST_FILENAME
    path.write_text(
        json.dumps(
            {
                **contract,
                "status": ablation.SELECTION_HPO_COMPLETED_STATUS,
                "selected_features": {
                    "long": [*ablation.ANCHORS, "context_a"],
                    "short": [*ablation.ANCHORS, "context_b"],
                },
                "hpo_params": {
                    "long": {"max_depth": 5},
                    "short": {"max_depth": 4},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_staged_selection_hpo_reuses_only_an_exact_sibling_contract(tmp_path):
    expected = _selection_hpo_contract(tmp_path / "inputs")
    reports = tmp_path / "reports"
    manifest = _write_completed_selection_hpo_manifest(reports / "prior", expected)

    reused, provenance = ablation._find_reusable_selection_hpo_manifest(
        expected,
        output_dir=reports / "new_run",
        manifest_path=None,
        force=False,
    )

    assert reused is not None
    assert reused["fingerprint"] == expected["fingerprint"]
    assert provenance["reused"] is True
    assert provenance["path"] == str(manifest)

    changed = _selection_hpo_contract(
        tmp_path / "inputs", candidate_context=["context_a", "different_context"]
    )
    no_reuse, mismatch = ablation._find_reusable_selection_hpo_manifest(
        changed,
        output_dir=reports / "other_run",
        manifest_path=None,
        force=False,
    )
    assert no_reuse is None
    assert mismatch["mismatched_candidates"] == 1

    with pytest.raises(ValueError, match="fingerprint does not match"):
        ablation._find_reusable_selection_hpo_manifest(
            changed,
            output_dir=reports / "other_run",
            manifest_path=manifest,
            force=False,
        )


def test_staged_selection_hpo_force_skips_exact_contract_and_rejects_legacy(tmp_path):
    expected = _selection_hpo_contract(tmp_path / "inputs")
    reports = tmp_path / "reports"
    _write_completed_selection_hpo_manifest(reports / "prior", expected)
    legacy = reports / "legacy" / ablation.SELECTION_HPO_MANIFEST_FILENAME
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "selected_features": {"long": ["old"], "short": ["old"]},
                "hpo_params": {"long": {"max_depth": 3}, "short": {"max_depth": 3}},
            }
        ),
        encoding="utf-8",
    )

    rerun, forced = ablation._find_reusable_selection_hpo_manifest(
        expected,
        output_dir=reports / "new_run",
        manifest_path=None,
        force=True,
    )
    assert rerun is None
    assert forced == {"mode": "forced_rerun", "reused": False}

    with pytest.raises(ValueError, match="strict reusable contract fields"):
        ablation._find_reusable_selection_hpo_manifest(
            expected,
            output_dir=reports / "new_run",
            manifest_path=legacy,
            force=False,
        )

    tampered = _write_completed_selection_hpo_manifest(
        reports / "tampered", expected
    )
    tampered_payload = json.loads(tampered.read_text(encoding="utf-8"))
    tampered_payload["fingerprint_inputs"]["candidate_context"] = ["tampered"]
    tampered.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint does not match its inputs"):
        ablation._read_completed_selection_hpo_manifest(tampered)

    legacy_only = tmp_path / "legacy_only_reports"
    legacy_only_manifest = legacy_only / "prior" / ablation.SELECTION_HPO_MANIFEST_FILENAME
    legacy_only_manifest.parent.mkdir(parents=True)
    legacy_only_manifest.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
    automatic, provenance = ablation._find_reusable_selection_hpo_manifest(
        expected,
        output_dir=legacy_only / "new_run",
        manifest_path=None,
        force=False,
    )
    assert automatic is None
    assert provenance["legacy_or_invalid_candidates"] == 1


def test_fixed_hpo_manifest_reuses_params_without_reusing_features(tmp_path):
    path = tmp_path / "selection_manifest.json"
    path.write_text(
        json.dumps(
            {
                "selected_features": {"long": ["stale_long"], "short": ["stale_short"]},
                "hpo_params": {
                    "long": {"max_depth": 5, "reg_lambda": 10.0},
                    "short": {"max_depth": 3, "reg_lambda": 4.0},
                },
            }
        ),
        encoding="utf-8",
    )

    params = ablation._load_fixed_hpo_params_manifest(path)

    assert params == {
        "long": {"max_depth": 5, "reg_lambda": 10.0},
        "short": {"max_depth": 3, "reg_lambda": 4.0},
    }
    assert "selected_features" not in params


def test_fixed_hpo_manifest_requires_both_sides(tmp_path):
    path = tmp_path / "selection_manifest.json"
    path.write_text(
        json.dumps({"hpo_params": {"long": {"max_depth": 5}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="short params"):
        ablation._load_fixed_hpo_params_manifest(path)


def test_staged_meta_selector_receives_complete_ae_gmm_feature_registry():
    assert tuple(ablation.AEGMM_CANDIDATES) == tuple(AE_GMM_FEATURE_COLUMNS)
    assert "gmm_cluster_posterior_7" in ablation.AEGMM_CANDIDATES
    assert "gmm_unknown_probability" in ablation.AEGMM_CANDIDATES
    assert "gmm_ood_score" in ablation.AEGMM_CANDIDATES
    assert "dae_b16_15" in ablation.AEGMM_CANDIDATES


def test_feature_store_augmentation_preserves_utc_timestamps(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features" / "20260301_000000"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "symbol=BTC.parquet"
    pd.DataFrame(
        {
            "fresh_context": [1.25],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")], name="ts"
        ),
    ).to_parquet(feature_path)
    monkeypatch.setattr(
        ablation,
        "_feature_file_for_symbol",
        lambda _feature_dir, _symbol: feature_path,
    )
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")],
            "__symbol__": ["BTC"],
            "side_name": ["long"],
        }
    )

    augmented = ablation._augment_from_feature_store(
        frame,
        feature_dir,
        ["fresh_context"],
    )

    assert str(augmented["__ts__"].dtype) == "datetime64[ns, UTC]"
    assert augmented.loc[0, "fresh_context"] == 1.25


def test_feature_store_augmentation_reads_static_only_columns(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features" / "20260301_000000"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "symbol=BTC.parquet"
    pd.DataFrame(
        {"physical_only": [1.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")], name="ts"
        ),
    ).to_parquet(feature_path)
    monkeypatch.setattr(
        ablation,
        "_feature_file_for_symbol",
        lambda _feature_dir, _symbol: feature_path,
    )
    calls = []

    def _logical_reader(**kwargs):
        calls.append(kwargs)
        return {
            "BTC": pd.DataFrame(
                {
                    "__ts__": [pd.Timestamp("2026-03-01 00:00:00")],
                    "__symbol__": ["BTC"],
                    "static_only": [2.5],
                }
            )
        }

    monkeypatch.setattr(
        ablation, "_read_feature_store_symbol_context_batch", _logical_reader
    )
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")],
            "__symbol__": ["BTC"],
            "side_name": ["long"],
        }
    )

    augmented = ablation._augment_from_feature_store(
        frame, feature_dir, ["physical_only", "static_only"]
    )

    assert len(calls) == 1
    assert calls[0]["columns"] == ["static_only"]
    assert augmented.loc[0, "physical_only"] == 1.0
    assert augmented.loc[0, "static_only"] == 2.5
    assert str(augmented["__ts__"].dtype) == "datetime64[ns, UTC]"


def test_feature_store_augmentation_prefers_logical_repair_over_stale_null(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features" / "20260301_000000"
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "symbol=BTC.parquet"
    pd.DataFrame(
        {"market_state": [float("nan")]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")], name="ts"
        ),
    ).to_parquet(feature_path)
    monkeypatch.setattr(
        ablation,
        "_feature_file_for_symbol",
        lambda _feature_dir, _symbol: feature_path,
    )
    monkeypatch.setattr(
        ablation,
        "_read_feature_store_symbol_context_batch",
        lambda **_kwargs: {
            "BTC": pd.DataFrame(
                {
                    "__ts__": [pd.Timestamp("2026-03-01 00:00:00")],
                    "__symbol__": ["BTC"],
                    "market_state": [3.5],
                }
            )
        },
    )
    frame = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-03-01 00:00:00", tz="UTC")],
            "__symbol__": ["BTC"],
            "side_name": ["long"],
        }
    )

    augmented = ablation._augment_from_feature_store(frame, feature_dir, ["market_state"])

    assert augmented.loc[0, "market_state"] == 3.5


def test_control_comparison_normalizes_week_key_dtype(tmp_path):
    score_col = "score_base_ev_residual_expert_hier_mapped"
    current = pd.DataFrame(
        {
            "scope": ["week"],
            "model": [score_col],
            "calendar_month": [None],
            "week_start": [pd.Timestamp("2026-04-06", tz="UTC")],
            "side_name": [None],
            "archetype_policy_key": [None],
            "selected_rows": [10.0],
            "mean_ev_after_1pct": [0.02],
        }
    )
    pd.DataFrame(
        {
            "scope": ["week"],
            "model": [score_col],
            "calendar_month": [None],
            "week_start": ["2026-04-06 00:00:00+00:00"],
            "side_name": [None],
            "archetype_policy_key": [None],
            "selected_rows": [10.0],
            "mean_ev_after_1pct": [0.01],
        }
    ).to_csv(tmp_path / "metrics.csv", index=False)

    comparison = ablation._compare_control_metrics(
        current,
        score_col=score_col,
        control_dir=tmp_path,
    )

    assert len(comparison) == 1
    assert comparison.loc[0, "delta_mean_ev_after_1pct"] == 0.01


def test_control_comparison_recomputes_on_exact_oos_rows(tmp_path):
    score_col = "score_base_ev_residual_expert_hier_mapped"
    timestamps = pd.to_datetime(
        ["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"], utc=True
    )
    current_oos = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["BTC", "ETH"],
            "side_name": ["long", "short"],
            "ev_after_1pct": [0.03, -0.01],
            "clean_exec": [1.0, 0.0],
            "dirty_positive": [0.0, 0.0],
            "full_path_bad_mae_1r": [0.0, 1.0],
            "timeout": [0.0, 0.0],
            "calendar_month": ["2026-04", "2026-04"],
            "week_start": [timestamps[0].normalize()] * 2,
            "archetype_policy_key": ["a", "b"],
            score_col: [0.9, 0.1],
        }
    )
    pd.DataFrame(
        {
            "__ts__": [*timestamps, pd.Timestamp("2026-03-31T23:00:00Z")],
            "__symbol__": ["BTC", "ETH", "SOL"],
            "side_name": ["long", "short", "long"],
            score_col: [0.1, 0.9, 1.0],
        }
    ).to_parquet(tmp_path / "oos_predictions.parquet", index=False)
    metrics = ablation._breakdown(current_oos, [score_col])

    comparison = ablation._compare_control_metrics(
        metrics,
        score_col=score_col,
        control_dir=tmp_path,
        current_oos=current_oos,
    )

    assert not comparison.empty
    assert set(comparison["comparison_basis"]) == {
        "exact_timestamp_symbol_side_overlap"
    }


def test_control_comparison_allows_partial_overlap_and_reranks_both_scores(tmp_path):
    score_col = "score_base_ev_residual_expert_hier_mapped"
    timestamps = pd.date_range("2026-04-01", periods=20, freq="h", tz="UTC")
    current_oos = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": [f"S{i}" for i in range(20)],
            "side_name": ["long"] * 20,
            "ev_after_1pct": [0.10] + [0.0] * 19,
            "clean_exec": [1.0] + [0.0] * 19,
            "dirty_positive": [0.0] * 20,
            "full_path_bad_mae_1r": [0.0] * 20,
            "timeout": [0.0] * 20,
            "calendar_month": ["2026-04"] * 20,
            "week_start": [timestamps[0].normalize()] * 20,
            "archetype_policy_key": ["a"] * 20,
            score_col: [1.0] + [0.0] * 19,
        }
    )
    # The control covers only ten rows and ranks the profitable row last.
    pd.DataFrame(
        {
            "__ts__": timestamps[:10],
            "__symbol__": [f"S{i}" for i in range(10)],
            "side_name": ["long"] * 10,
            score_col: [0.0] + [1.0] * 9,
        }
    ).to_parquet(tmp_path / "oos_predictions.parquet", index=False)
    metrics = ablation._breakdown(current_oos, [score_col])

    comparison = ablation._compare_control_metrics(
        metrics,
        score_col=score_col,
        control_dir=tmp_path,
        current_oos=current_oos,
    )
    overall = comparison.loc[comparison["scope"].eq("overall")].iloc[0]

    assert overall["comparison_current_rows"] == 20
    assert overall["comparison_overlap_rows"] == 10
    assert overall["comparison_missing_control_rows"] == 10
    assert overall["mean_ev_after_1pct_ablation"] == 0.10
    assert overall["mean_ev_after_1pct_control"] == 0.0


def test_resolved_train_before_purges_overlapping_forward_paths():
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-03-30T00:00:00Z", "2026-03-31T23:00:00Z"], utc=True
            ),
            "__label_path_end_ts__": pd.to_datetime(
                ["2026-03-31T00:00:00Z", "2026-04-01T23:00:00Z"], utc=True
            ),
        }
    )

    train = ablation._resolved_train_before(
        frame, pd.Timestamp("2026-04-01T00:00:00Z")
    )

    assert train.index.tolist() == [0]


def test_resolved_train_before_fails_closed_without_resolution_column():
    with pytest.raises(ValueError, match="__label_path_end_ts__"):
        ablation._resolved_train_before(
            pd.DataFrame(
                {"__ts__": pd.to_datetime(["2026-03-30T00:00:00Z"], utc=True)}
            ),
            pd.Timestamp("2026-04-01T00:00:00Z"),
        )


def test_current_handoff_loader_preserves_utc_label_resolution_timestamps(tmp_path):
    ts = pd.Timestamp("2026-07-01T00:00:00Z")
    path_end = ts + pd.Timedelta(hours=25)
    handoff = tmp_path / "handoff.parquet"
    ledger = tmp_path / "ledger.parquet"
    pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "__label_path_end_ts__": [path_end],
            "score": [0.5],
            "base_margin_to_cutoff": [0.1],
            "base_margin_to_cutoff_z": [0.2],
            "base_signal_zscore_within_archetype": [0.3],
            "archetype_policy_key": ["long_mixed"],
            "archetype_label_family": ["mixed"],
        }
    ).to_parquet(handoff, index=False)
    pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "ev_after_1pct": [0.01],
            "clean_exec": [1.0],
            "dirty_positive": [0.0],
            "full_path_bad_mae_1r": [0.0],
            "timeout": [0.0],
        }
    ).to_parquet(ledger, index=False)

    frame = ablation._load_current_handoff(
        handoff,
        ledger,
        [],
        end_exclusive=pd.Timestamp("2026-07-02T00:00:00Z"),
    )

    assert frame.loc[0, "__ts__"] == ts
    assert frame.loc[0, "__label_path_end_ts__"] == path_end
