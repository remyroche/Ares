import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_path_auxiliary_lgbm_models as runner


def _labels() -> pd.DataFrame:
    rows = 12
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"] * (rows // 2),
            "side": [1, -1] * (rows // 2),
            "archetype": ["trend", "mean_revert", "trend"] * 4,
            "selected_top40": [True] * rows,
            "__label_end_ts__": pd.date_range(
                "2026-01-01 01:00", periods=rows, freq="h", tz="UTC"
            ),
            "__log1p_time_to_first_meaningful_mfe_hours_12h__": np.linspace(
                0.2, 1.2, rows
            ),
            "__log1p_peak_mfe_atr_12h__": np.linspace(0.1, 0.8, rows),
            "__log1p_mae_before_meaningful_mfe_atr_12h__": np.linspace(0.1, 0.7, rows),
            "__log1p_bars_before_price_stops_decreasing_12h__": np.linspace(
                0.0, 1.0, rows
            ),
            "__log1p_future_slope_atr_per_hour_12h__": np.linspace(0.05, 0.9, rows),
        }
    )
    for column in runner.ALL_SUPPORTIVE_LABEL_COLUMNS:
        frame[column] = np.float32(0.0)
    frame["gmm_representation_available"] = (
        np.arange(len(frame), dtype=np.int8) % 2
    ).astype(np.float32)
    frame["candidate_id"] = runner.candidate_id_series(
        frame["__ts__"], frame["__symbol__"], "1h", frame["side"]
    ).to_numpy()
    return frame


def _add_candidate_model_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values = np.linspace(0.1, 0.9, len(out), dtype=np.float32)
    for offset, column in enumerate(runner.MANDATORY_HANDOFF_MODEL_FEATURES):
        out[column] = values + np.float32(offset)
    out["gmm_representation_available"] = (
        np.arange(len(out), dtype=np.int8) % 2
    ).astype(np.float32)
    out["selected_top40"] = True
    out["candidate_id"] = runner.candidate_id_series(
        out["__ts__"],
        out["__symbol__"],
        "1h",
        out["side"] if "side" in out else out["side_name"],
    ).to_numpy()
    return out


def test_runner_persists_side_bundles_oof_and_availability(monkeypatch, tmp_path):
    labels = _labels()
    seen = {}

    def fake_load_labels(*_args, **kwargs):
        frame = labels.copy()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["side"] = runner._normalize_side(frame["side"])
        for column in runner.CONTEXT_COLUMNS:
            if column not in frame:
                frame[column] = "unknown"
        return frame, {"rows_after_identity_and_caps": len(frame)}

    def fake_static_columns(_feature_dir, symbols):
        assert set(symbols) == {"BTC/USD:USD", "ETH/USD:USD"}
        return ["f_a", "f_b"]

    def fake_universe(columns):
        selected = [
            column
            for column in columns
            if column in ("f_a", "f_b", "gmm_representation_available")
            or column.startswith("base_archetype_label__")
        ]
        return selected, {"available_selected_features": selected, "contract": "test"}

    def fake_static(frame, **kwargs):
        seen["static_index"] = frame[["__ts__", "__symbol__", "side"]].copy()
        seen["requested"] = kwargs["requested_features"]
        return pd.DataFrame(
            {"f_a": np.arange(len(frame)), "f_b": 1.0}, index=frame.index
        ), {
            "reader": "test",
            "available_feature_names": ["f_a", "f_b"],
            "missing_features": [],
        }

    def fake_select(X, y, **kwargs):
        seen["selector_calls"] = seen.get("selector_calls", 0) + 1
        assert {"f_a", "f_b"}.issubset(X.columns)
        assert any(column.startswith("base_archetype_label__") for column in X)
        assert set(kwargs["sides"]) == {"long", "short"}
        mandatory = kwargs["mandatory_features_by_side"]
        assert all(mandatory[side] for side in ("long", "short"))
        return {
            "selected_features_by_side": {
                "long": ["f_a", *mandatory["long"]],
                "short": ["f_b", *mandatory["short"]],
            },
            "selection_metrics": {},
        }

    def fake_fit(X, y, **kwargs):
        seen.setdefault("preset_hpo_calls", []).append(
            kwargs.get("preset_hpo_params_by_side")
        )
        assert kwargs["n_trials"] == 2
        assert kwargs["hpo_rows"] == 45_000
        assert np.all(
            (kwargs["sample_weight"] >= 0.5) & (kwargs["sample_weight"] <= 2.0)
        )
        decision = pd.to_datetime(kwargs["timestamps"], utc=True)
        reference_end = pd.Timestamp(kwargs["selection_hpo_reference_end"])
        oof = np.full(len(y), np.nan, dtype=np.float32)
        oof[decision >= reference_end] = (
            np.asarray(y, dtype=np.float32)[decision >= reference_end] + 0.01
        )
        selected = kwargs["selected_features_by_side"]
        return {
            "oof_predictions": oof,
            "oof_fold_ids": np.where(np.isfinite(oof), 0, -1).astype(np.int16),
            "purge_hours": kwargs["purge_hours"],
            "models_by_side": {
                "long": {
                    "selected_features": selected["long"],
                    "best_params": {"n_estimators": 1},
                    "model": {"side": "long"},
                    "fold_metrics": [
                        {
                            "fold": 0,
                            "train_end": "2026-01-01T04:00:00Z",
                            "valid_start": "2026-01-01T06:00:00Z",
                            "valid_end": "2026-01-01T11:00:00Z",
                        }
                    ],
                },
                "short": {
                    "selected_features": selected["short"],
                    "best_params": {"n_estimators": 1},
                    "model": {"side": "short"},
                    "fold_metrics": [
                        {
                            "fold": 0,
                            "train_end": "2026-01-01T03:00:00Z",
                            "valid_start": "2026-01-01T06:00:00Z",
                            "valid_end": "2026-01-01T11:00:00Z",
                        }
                    ],
                },
            },
        }

    monkeypatch.setattr(runner, "_load_labels", fake_load_labels)
    monkeypatch.setattr(runner, "_static_feature_columns", fake_static_columns)
    monkeypatch.setattr(runner, "configured_auxiliary_feature_universe", fake_universe)
    monkeypatch.setattr(runner, "_load_static_features", fake_static)
    monkeypatch.setattr(runner, "select_features_with_current_pipeline", fake_select)
    monkeypatch.setattr(runner, "fit_side_aware_auxiliary_models", fake_fit)

    output = tmp_path / "output"
    manifest = runner.run(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features" / "20260101_000000",
        output_dir=output,
        n_trials=2,
        max_rows=5,
        labels_are_canonical_top40=True,
        selection_hpo_reference_end="2026-01-01T06:00:00Z",
    )

    assert manifest["heads"].keys() == {
        "time_to_first_meaningful_mfe",
        "peak_mfe_12h_atr",
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    }
    assert manifest["hpo_rows_per_side"] == 45_000
    reference_contract = manifest["selection_hpo_reference_contract"]
    assert (
        reference_contract["selection_hpo_reference_end"] == "2026-01-01T06:00:00+00:00"
    )
    assert (
        reference_contract["contract_sha256"]
        == manifest["selection_hpo_reference_contract_sha256"]
    )
    assert (
        manifest["base_archetype_label_feature_contract"]["canonical_source"]
        == "archetype"
    )
    assert {"f_a", "f_b"}.issubset(seen["requested"])
    assert any(
        feature.startswith("base_archetype_label__") for feature in seen["requested"]
    )
    assert seen["static_index"]["__ts__"].dt.tz is not None
    for target in runner.TARGET_COLUMNS:
        assert (output / target / "oof_predictions.parquet").exists()
        oof_frame = pd.read_parquet(output / target / "oof_predictions.parquet")
        assert {
            "candidate_id",
            "oof_prediction_log1p",
            "oof_fold",
            "available_at",
            "validation_start",
            "train_decision_cutoff",
            "label_resolution_available_at",
        }.issubset(oof_frame.columns)
        available = oof_frame["oof_available"].astype(bool)
        assert (
            oof_frame.loc[available, "__ts__"] >= pd.Timestamp("2026-01-01T06:00:00Z")
        ).all()
        assert (
            oof_frame["oof_after_selection_hpo_reference_end"]
            .eq(
                (oof_frame["__ts__"] >= pd.Timestamp("2026-01-01T06:00:00Z")).astype(
                    int
                )
            )
            .all()
        )
        assert oof_frame.loc[available, "candidate_id"].notna().all()
        assert (
            oof_frame.loc[available, "label_resolution_available_at"]
            <= oof_frame.loc[available, "train_decision_cutoff"]
        ).all()
        assert (
            oof_frame.loc[available, "train_decision_cutoff"]
            < oof_frame.loc[available, "validation_start"]
        ).all()
        source_manifest = json.loads(
            (output / target / "oof_predictions.manifest.json").read_text()
        )
        assert source_manifest["prediction_role"] == runner.PREDICTION_ROLES[target]
        assert len(source_manifest["prediction_role_manifest_sha256"]) == 64
        assert (output / target / "params_by_side.json").exists()
        assert (output / target / "bundles" / "long.joblib").exists()
        assert (output / target / "bundles" / "short.joblib").exists()
        long_bundle = __import__("joblib").load(
            output / target / "bundles" / "long.joblib"
        )
        assert (
            long_bundle["model_role"]
            == "all_resolved_final_inference_excluded_from_oos_metrics"
        )
        assert "final_inference_model" in long_bundle
        assert long_bundle["base_archetype_label_feature_contract"][
            "canonical_features"
        ]
        assert any(
            feature.startswith("base_archetype_label__")
            for feature in long_bundle["selected_features"]
        )
        metrics = json.loads((output / target / "metrics.json").read_text())
        assert set(metrics["by_side"]) == {"long", "short"}
        assert set(metrics["by_archetype"]) == {"trend", "mean_revert"}
        assert set(metrics["by_representation_availability"]) == {
            "available",
            "missing",
        }
        assert set(metrics["by_side_representation_availability"]) == {
            "long",
            "short",
        }
    timing = pd.read_parquet(
        output / "time_to_first_meaningful_mfe" / "oof_predictions.parquet"
    )
    peak = pd.read_parquet(output / "peak_mfe_12h_atr" / "oof_predictions.parquet")
    mae = pd.read_parquet(
        output / "mae_before_meaningful_mfe_atr" / "oof_predictions.parquet"
    )
    turning = pd.read_parquet(
        output / "bars_before_price_stops_decreasing" / "oof_predictions.parquet"
    )
    slope = pd.read_parquet(
        output / "future_slope_atr_per_hour" / "oof_predictions.parquet"
    )
    assert "pred_time_to_first_meaningful_mfe_12h" in timing
    assert "pred_peak_mfe_12h_atr" in peak
    assert "pred_mae_before_meaningful_mfe_atr_12h" in mae
    assert "pred_bars_before_price_stops_decreasing_12h" in turning
    assert "pred_future_slope_atr_per_hour_12h" in slope
    assert json.loads((output / "input_universe_availability.json").read_text())[
        "exact_alignment"
    ].startswith("static feature")
    assert seen["selector_calls"] == len(runner.TARGET_COLUMNS)

    reused_output = tmp_path / "reused_output"
    reused_manifest = runner.run(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features" / "20260101_000000",
        output_dir=reused_output,
        n_trials=2,
        max_rows=5,
        labels_are_canonical_top40=True,
        selection_hpo_reference_end="2026-01-01T06:00:00Z",
    )
    assert seen["selector_calls"] == len(runner.TARGET_COLUMNS)
    assert all(
        value is not None
        for value in seen["preset_hpo_calls"][-len(runner.TARGET_COLUMNS) :]
    )
    assert reused_manifest["selection_hpo_reuse"]["auto_reused"] is True
    assert reused_manifest["selection_hpo_reuse"]["source_manifest"].endswith(
        "/output/manifest.json"
    )
    forced_output = tmp_path / "forced_output"
    forced_manifest = runner.run(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features" / "20260101_000000",
        output_dir=forced_output,
        n_trials=2,
        max_rows=5,
        labels_are_canonical_top40=True,
        selection_hpo_reference_end="2026-01-01T06:00:00Z",
        force_selection_hpo=True,
    )
    assert seen["selector_calls"] == 2 * len(runner.TARGET_COLUMNS)
    assert (
        forced_manifest["selection_hpo_reuse"]["reason"]
        == "force_selection_hpo_requested"
    )

    # Simulate an interruption after every head was atomically persisted but
    # before the root manifest became visible.  The exact checkpoint must avoid
    # rerunning selection, HPO, OOF folds, or final inference fits.
    (output / "manifest.json").unlink()
    selector_calls_before_resume = seen["selector_calls"]
    fit_calls_before_resume = len(seen["preset_hpo_calls"])
    resumed_manifest = runner.run(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features" / "20260101_000000",
        output_dir=output,
        n_trials=2,
        max_rows=5,
        labels_are_canonical_top40=True,
        selection_hpo_reference_end="2026-01-01T06:00:00Z",
    )
    assert resumed_manifest["heads"] == manifest["heads"]
    assert seen["selector_calls"] == selector_calls_before_resume
    assert len(seen["preset_hpo_calls"]) == fit_calls_before_resume
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    assert set(checkpoint["heads"]) == set(runner.TARGET_COLUMNS)
    assert all(
        "complete" in checkpoint["heads"][target] for target in runner.TARGET_COLUMNS
    )


def test_runner_refuses_to_overwrite_nonempty_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="--overwrite"):
        runner.run(
            labels_path=tmp_path / "labels",
            feature_dir=tmp_path / "features" / "20260101_000000",
            output_dir=output,
            selection_hpo_reference_end="2026-01-01T00:00:00Z",
        )


def test_archetype_context_join_is_exact_utc_and_candidate_scoped(tmp_path):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1, 4, 5], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context = _add_candidate_model_context(context)
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    joined, report = runner._join_archetype_context(labels, path)

    assert len(joined) == 4
    assert report["rows_unmatched"] == 8
    assert report["selected_archetype_column"] == "archetype_label_family"
    assert set(runner._archetype_context(joined)) == {"trend", "mean_revert"}


def test_capped_context_pushdown_compares_utc_epoch_not_host_wall_clock(tmp_path):
    labels = _labels().iloc[:4].copy()
    labels["__ts__"] = pd.to_datetime(
        [
            "2026-03-29T00:00:00Z",
            "2026-03-29T01:00:00Z",
            "2026-10-25T00:00:00Z",
            "2026-10-25T01:00:00Z",
        ],
        utc=True,
    )
    labels["candidate_id"] = runner.candidate_id_series(
        labels["__ts__"], labels["__symbol__"], "1h", labels["side"]
    ).to_numpy()
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels[["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context = _add_candidate_model_context(context)
    # Historical stores persist UTC as naive parquet timestamps.
    context["__ts__"] = context["__ts__"].dt.tz_localize(None)
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    wanted = list(context.columns)
    actual, contract, _ = runner._read_context_for_label_keys(
        path,
        wanted,
        labels,
        timestamp_column="__ts__",
        symbol_column="__symbol__",
        side_column="side_name",
    )

    assert contract == "duckdb_exact_candidate_key_pushdown"
    actual_ts = pd.to_datetime(actual["__ts__"], utc=True)
    assert actual_ts.tolist() == labels["__ts__"].tolist()


def test_legacy_context_derives_canonical_candidate_ids_and_matches_labels(tmp_path):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1, 4, 5], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context = _add_candidate_model_context(context).drop(columns="candidate_id")
    path = tmp_path / "legacy_context.parquet"
    context.to_parquet(path, index=False)

    joined, _ = runner._join_archetype_context(labels, path)

    expected = runner.candidate_id_series(
        joined["__ts__"], joined["__symbol__"], "1h", joined["side"]
    ).tolist()
    assert joined["candidate_id"].tolist() == expected


def test_archetype_context_filters_selected_top40_before_join_and_records_hash(
    tmp_path,
):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1, 4, 5], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context = _add_candidate_model_context(context)
    # A duplicate rejected row proves the selected-top40 filter runs before
    # duplicate-key validation and the UTC inner join.
    rejected_duplicate = context.iloc[[0]].copy()
    rejected_duplicate["selected_top40"] = False
    context = pd.concat([context, rejected_duplicate], ignore_index=True)
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    joined, report = runner._join_archetype_context(labels, path)

    assert len(joined) == 4
    assert report["rows_source"] == 5
    assert report["rows_selected_before_identity_validation"] == 4
    assert report["rows_filtered_out"] == 1
    assert report["selected_population_identity_sha256"]


def test_archetype_context_rejects_ambiguous_selected_top40_values(tmp_path):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"]
    context = _add_candidate_model_context(context)
    context["selected_top40"] = context["selected_top40"].astype(str)
    context.loc[0, "selected_top40"] = "yes"
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="non-boolean selected_top40"):
        runner._join_archetype_context(labels, path)


def test_labels_only_population_requires_explicit_canonical_flag():
    labels = _labels()
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])

    with pytest.raises(ValueError, match="--labels-are-canonical-top40"):
        runner._join_archetype_context(labels, None)

    joined, report = runner._join_archetype_context(
        labels, None, labels_are_canonical_top40=True
    )
    assert len(joined) == len(labels)
    assert report["selection_source"] == "labels"
    labels_without_flag = labels.drop(columns="selected_top40")
    joined, report = runner._join_archetype_context(
        labels_without_flag, None, labels_are_canonical_top40=True
    )
    assert len(joined) == len(labels_without_flag)
    assert report["selection_column"] == "declared_all_rows_canonical_top40"


def test_runner_requires_explicit_selection_hpo_reference_end(tmp_path):
    with pytest.raises(
        ValueError, match="selection_hpo_reference_end must be declared"
    ):
        runner.run(
            labels_path=tmp_path / "labels",
            feature_dir=tmp_path / "features" / "20260101_000000",
            output_dir=tmp_path / "output",
        )


def test_handoff_generated_model_context_is_joined_and_overlays_missing_static_feature(
    monkeypatch, tmp_path
):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1, 4, 5], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context["gmm_prob_0"] = [0.9, 0.2, 0.8, 0.3]
    context = _add_candidate_model_context(context)
    path = tmp_path / "context.parquet"
    context.to_parquet(path, index=False)

    def fake_universe(columns):
        selected = [column for column in ("gmm_prob_0",) if column in columns]
        return selected, {"available_selected_features": selected, "contract": "test"}

    monkeypatch.setattr(runner, "configured_auxiliary_feature_universe", fake_universe)
    joined, report = runner._join_archetype_context(labels, path)
    assert report["handoff_model_feature_columns"] == [
        *runner.MANDATORY_HANDOFF_MODEL_FEATURES,
        "gmm_prob_0",
    ]

    matrix = pd.DataFrame(
        {"static_x": np.arange(len(joined), dtype=np.float32)}, index=joined.index
    )
    overlaid, availability = runner._overlay_handoff_model_features(
        matrix,
        joined,
        requested_features=["static_x", "gmm_prob_0"],
        static_report={
            "available_feature_names": ["static_x"],
            "available_features": 1,
            "missing_features": ["gmm_prob_0"],
        },
        handoff_feature_columns=report["handoff_model_feature_columns"],
    )
    np.testing.assert_allclose(overlaid["gmm_prob_0"], [0.9, 0.2, 0.8, 0.3])
    assert availability["handoff_overlay_features"] == ["gmm_prob_0"]
    assert availability["missing_features"] == []


def test_archetype_context_join_rejects_duplicate_keys(tmp_path):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 0, 1], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "trend", "mean_revert"]
    context = _add_candidate_model_context(context)
    path = tmp_path / "duplicate_context.parquet"
    context.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="duplicate UTC context keys"):
        runner._join_archetype_context(labels, path)


def test_missing_archetype_context_fails_instead_of_using_unknown():
    labels = _labels().drop(columns="archetype")
    labels["side"] = runner._normalize_side(labels["side"])
    for column in runner.CONTEXT_COLUMNS:
        if column not in labels:
            labels[column] = "unknown"

    with pytest.raises(ValueError, match="--archetype-context-path"):
        runner._join_archetype_context(labels, None)


def test_archetype_context_join_requires_finite_candidate_model_context(tmp_path):
    labels = _labels().drop(columns="archetype")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    labels["side"] = runner._normalize_side(labels["side"])
    context = labels.loc[[0, 1, 4, 5], ["__ts__", "__symbol__", "side"]].rename(
        columns={"side": "side_name"}
    )
    context["archetype_label_family"] = ["trend", "mean_revert"] * 2
    context = _add_candidate_model_context(context)
    context.loc[0, "score"] = np.nan
    path = tmp_path / "invalid_context.parquet"
    context.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="must be finite on every handoff row"):
        runner._join_archetype_context(labels, path)


def test_static_feature_load_uses_canonical_store_and_exact_row_identity(tmp_path):
    from extreme_price_movements.static_feature_store import append_static_features

    store_ts = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    index = pd.date_range("2025-12-31 22:00", periods=4, freq="h", tz="UTC")
    symbols = ["BTC/USD:USD", "ETH/USD:USD"]
    append_static_features(
        {
            "ret1h": pd.DataFrame(
                [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]],
                index=index,
                columns=symbols,
                dtype=np.float32,
            ),
            "mkt_rv_ratio": pd.DataFrame(
                [[21.0, 31.0], [22.0, 32.0], [23.0, 33.0], [24.0, 34.0]],
                index=index,
                columns=symbols,
                dtype=np.float32,
            ),
        },
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index,
        columns=symbols,
        source="pipeline",
    )
    rows = pd.DataFrame(
        {
            "__ts__": [index[2], index[0], index[3]],
            "__symbol__": [symbols[1], symbols[0], symbols[0]],
            "side": ["short", "long", "long"],
        }
    )
    loaded, report = runner._load_static_features(
        rows,
        feature_dir=tmp_path / "features" / "20260101_000000",
        requested_features=["ret1h", "mkt_rv_ratio"],
    )
    np.testing.assert_allclose(loaded["ret1h"], [13.0, 1.0, 4.0])
    np.testing.assert_allclose(loaded["mkt_rv_ratio"], [33.0, 21.0, 24.0])
    assert report["reader"].endswith("read_static_features")
    assert report["missing_features"] == []


def test_static_feature_load_allows_symbol_local_missing_columns(monkeypatch, tmp_path):
    from extreme_price_movements import static_feature_store

    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": [index[0], index[0]],
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": ["long", "short"],
        }
    )

    class Loaded:
        def __contains__(self, key):
            return key in {"shared", "btc_only"}

        def symbol_frame(self, symbol, *, keys):
            data = {"shared": [1.0 if symbol.startswith("BTC") else 2.0]}
            if symbol.startswith("BTC"):
                data["btc_only"] = [3.0]
            return pd.DataFrame(data, index=[index[0]])

    monkeypatch.setattr(
        static_feature_store, "read_static_features", lambda **_kwargs: Loaded()
    )
    loaded, _ = runner._load_static_features(
        rows,
        feature_dir=tmp_path / "features" / "20260101_000000",
        requested_features=["shared", "btc_only"],
    )

    np.testing.assert_allclose(loaded["shared"], [1.0, 2.0])
    assert loaded.loc[0, "btc_only"] == 3.0
    assert np.isnan(loaded.loc[1, "btc_only"])


def test_static_feature_load_reuses_bme_reader_buffer_for_full_selected_load(
    monkeypatch, tmp_path
):
    from extreme_price_movements import static_feature_store

    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    values = {
        "BTC/USD:USD": pd.DataFrame(
            {"f_a": [1.0, 2.0, 3.0, 4.0], "f_b": [11.0, 12.0, 13.0, 14.0]},
            index=index,
            dtype=np.float32,
        ),
        "ETH/USD:USD": pd.DataFrame(
            {"f_a": [21.0, 22.0, 23.0, 24.0], "f_b": [31.0, 32.0, 33.0, 34.0]},
            index=index,
            dtype=np.float32,
        ),
    }
    calls = []

    class Loaded:
        __static_feature_cache_nbytes__ = 4_096

        def __init__(self, kwargs):
            self.keys = list(kwargs["feature_keys"])
            self.symbols = list(kwargs["symbols"])
            self.periods = kwargs.get("allowed_periods")

        def __contains__(self, key):
            return key in self.keys

        def symbol_frame(self, symbol, *, keys):
            frame = values[symbol].loc[:, list(keys)]
            if self.periods:
                mask = np.zeros(len(frame), dtype=bool)
                for start, end in self.periods:
                    mask |= (frame.index >= start) & (frame.index < end)
                frame = frame.loc[mask]
            return frame

    def fake_read(**kwargs):
        calls.append(kwargs)
        return Loaded(kwargs)

    monkeypatch.setattr(static_feature_store, "read_static_features", fake_read)
    cache = runner._StaticFeatureReadCache(max_bytes=8_192, max_entries=8)
    selection_rows = pd.DataFrame(
        {
            "__ts__": [index[0], index[2]],
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side": ["long", "short"],
        }
    )
    selected_rows = pd.DataFrame(
        {
            "__ts__": [index[0], index[1], index[2], index[3]],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD", "ETH/USD:USD", "ETH/USD:USD"],
            "side": ["long", "long", "short", "short"],
        }
    )
    feature_dir = tmp_path / "features" / "20260101_000000"

    selection, selection_report = runner._load_static_features(
        selection_rows,
        feature_dir=feature_dir,
        requested_features=["f_a", "f_b"],
        read_cache=cache,
    )
    full, full_report = runner._load_static_features(
        selected_rows,
        feature_dir=feature_dir,
        requested_features=["f_a"],
        read_cache=cache,
    )

    np.testing.assert_allclose(selection["f_a"], [1.0, 23.0])
    np.testing.assert_allclose(full["f_a"], [1.0, 2.0, 23.0, 24.0])
    assert len(calls) == 2
    assert "allowed_periods" not in calls[0]
    assert "allowed_periods" not in calls[1]
    assert selection_report["read_cache"]["admissions"] == 1
    assert full_report["read_cache"]["hits"] == 1
    assert full_report["read_cache"]["reused_rows"] == 3


def test_static_feature_cache_preserves_real_store_utc_rows(monkeypatch, tmp_path):
    from extreme_price_movements import static_feature_store
    from extreme_price_movements.static_feature_store import append_static_features

    store_ts = pd.Timestamp("2026-01-01", tz="UTC")
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    symbols = ["BTC/USD:USD", "ETH/USD:USD"]
    append_static_features(
        {
            "f_a": pd.DataFrame(
                [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]],
                index=index,
                columns=symbols,
                dtype=np.float32,
            ),
            "f_b": pd.DataFrame(
                [[21.0, 31.0], [22.0, 32.0], [23.0, 33.0], [24.0, 34.0]],
                index=index,
                columns=symbols,
                dtype=np.float32,
            ),
        },
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index,
        columns=symbols,
        source="pipeline",
    )
    calls = []
    original_read = static_feature_store.read_static_features

    def counted_read(**kwargs):
        calls.append(kwargs)
        return original_read(**kwargs)

    monkeypatch.setattr(static_feature_store, "read_static_features", counted_read)
    cache = runner._StaticFeatureReadCache(max_bytes=1_000_000, max_entries=8)
    selection_rows = pd.DataFrame(
        {
            "__ts__": [index[0], index[2]],
            "__symbol__": [symbols[0], symbols[1]],
            "side": ["long", "short"],
        }
    )
    full_rows = pd.DataFrame(
        {
            "__ts__": index,
            "__symbol__": [symbols[0], symbols[0], symbols[1], symbols[1]],
            "side": ["long", "long", "short", "short"],
        }
    )
    feature_dir = tmp_path / "features" / "20260101_000000"

    selection, selection_report = runner._load_static_features(
        selection_rows,
        feature_dir=feature_dir,
        requested_features=["f_a", "f_b"],
        read_cache=cache,
        sampled_periods=True,
    )
    loaded, report = runner._load_static_features(
        full_rows,
        feature_dir=feature_dir,
        requested_features=["f_a"],
        read_cache=cache,
    )

    np.testing.assert_allclose(selection["f_a"], [1.0, 13.0])
    np.testing.assert_allclose(loaded["f_a"], [1.0, 2.0, 13.0, 14.0])
    assert len(calls) == 2
    assert len(calls[0]["allowed_periods"]) == 2
    assert "allowed_periods" not in calls[1]
    assert selection_report["allowed_period_count"] == 2
    assert report["read_cache"]["hits"] == 1
    assert report["read_cache"]["retained_bytes"] <= 1_000_000


def test_sparse_bme_static_read_uses_coalesced_utc_periods(monkeypatch, tmp_path):
    from extreme_price_movements import static_feature_store

    index = pd.date_range("2026-01-01", periods=73, freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": [index[0], index[72]],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side": ["long", "long"],
        }
    )
    calls = []

    class Loaded:
        def __contains__(self, key):
            return key == "f_a"

        def symbol_frame(self, _symbol, *, keys):
            assert keys == ["f_a"]
            return pd.DataFrame(
                {"f_a": np.arange(len(index), dtype=np.float32)}, index=index
            )

    def fake_read(**kwargs):
        calls.append(kwargs)
        return Loaded()

    monkeypatch.setattr(static_feature_store, "read_static_features", fake_read)
    loaded, report = runner._load_static_features(
        rows,
        feature_dir=tmp_path / "features" / "20260101_000000",
        requested_features=["f_a"],
        sampled_periods=True,
    )

    np.testing.assert_allclose(loaded["f_a"], [0.0, 72.0])
    assert calls[0]["allowed_periods"] == [
        (index[0], index[0] + pd.Timedelta(hours=1)),
        (index[72], index[72] + pd.Timedelta(hours=1)),
    ]
    assert calls[0]["allowed_periods"] != [
        (index[0], index[72] + pd.Timedelta(nanoseconds=1))
    ]
    assert report["sampled_period_read"]
    assert report["allowed_period_count"] == 2


def test_static_feature_read_cache_refuses_entries_above_explicit_byte_cap():
    class Loaded:
        __static_feature_cache_nbytes__ = 4_097

    cache = runner._StaticFeatureReadCache(max_bytes=4_096, max_entries=8)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    cache.put(
        symbols=["BTC/USD:USD"],
        requested_features=["f_a"],
        periods=[(start, start + pd.Timedelta(hours=1))],
        loaded=Loaded(),
    )

    assert cache.report()["retained_entries"] == 0
    assert cache.report()["rejected_entries"] == 1


def test_runner_resource_preflight_happens_before_labels_load(monkeypatch, tmp_path):
    events = []

    class Guard:
        def preflight(self, stage):
            events.append(stage)

        def checkpoint(self, stage):
            events.append(stage)

    def fail_load(*_args, **_kwargs):
        assert events == ["labels_load"]
        raise RuntimeError("stop after preflight")

    monkeypatch.setattr(runner, "_load_labels", fail_load)
    with pytest.raises(RuntimeError, match="stop after preflight"):
        runner.run(
            labels_path=tmp_path / "labels",
            feature_dir=tmp_path / "features" / "20260101_000000",
            output_dir=tmp_path / "output",
            labels_are_canonical_top40=True,
            selection_hpo_reference_end="2026-01-01T06:00:00Z",
            resource_guard=Guard(),
        )
