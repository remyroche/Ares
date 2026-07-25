from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.path_archetype_labels import PATH_SHAPE_TYPES

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_execution_ev_meta", ROOT / "scripts" / "run_execution_ev_meta.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _frame(rows: int = 96) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, rows)
    alpha = x * 0.01
    probabilities = np.full((rows, len(PATH_SHAPE_TYPES)), 0.04)
    winners = np.where(x > 0.0, 1, 0)
    probabilities[np.arange(rows), winners] = 0.72
    catboost_entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    frame = pd.DataFrame(
        {
            "__ts__": times,
            "execution_decision_utc": times + pd.Timedelta(hours=1),
            "execution_label_end_utc": times + pd.Timedelta(hours=13),
            "__symbol__": np.where(np.arange(rows) % 2, "ETH/USD:USD", "BTC/USD:USD"),
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "candidate_id": [f"candidate-{row}" for row in range(rows)],
            "catboost_archetype": [PATH_SHAPE_TYPES[index] for index in winners],
            "execution_net_ev_12h": alpha + x * 0.004,
            "existing_alpha_ev": alpha,
            "execution_gross_ev_12h": alpha + x * 0.004 + 0.003,
            "execution_cost_return": 0.003,
            "execution_exit_hour": np.where(alpha + x * 0.004 > 0.0, 4, 12),
            "execution_exit_reason": np.where(
                alpha + x * 0.004 > 0.0, "trailing", "full_stop"
            ),
            "existing_alpha_ev_source_basis": alpha - 0.007,
            "alpha_source_cost_return": 0.01,
            "pred_time_to_mfe_12h": 5.0 - x,
            "pred_peak_mfe_12h": 0.02 + x * 0.001,
            "pred_mae_before_meaningful_mfe_atr": 0.8 - x * 0.1,
            "pred_bars_before_price_stops_decreasing": 5.0 - x,
            "pred_favorable_path_slope_atr_per_hour": 0.4 + x * 0.1,
            "catboost_entropy": catboost_entropy,
            "base_prediction_uncertainty": 0.2 + np.abs(x) * 0.1,
            "meta_leaf_support_log1p": 2.0 + x,
            "base_archetype_label__family__trend": (x > 0.0).astype(float),
            "available_at": times,
        }
    )
    for index in range(len(PATH_SHAPE_TYPES)):
        frame[f"catboost_p_{index}"] = probabilities[:, index]
    return frame


def _provenance(rows: int) -> dict[str, object]:
    features = {
        "catboost_archetype": (
            "predicted_path_archetype",
            "frozen CatBoost path classifier",
            False,
        ),
        "existing_alpha_ev": ("alpha_score", "frozen alpha EV", True),
        "pred_time_to_mfe_12h": ("time_to_mfe", "frozen OOF time head", True),
        "pred_peak_mfe_12h": ("peak_mfe", "frozen OOF peak head", True),
        "pred_mae_before_meaningful_mfe_atr": (
            "mae_before_meaningful_mfe",
            "frozen OOF adverse-depth head",
            True,
        ),
        "pred_bars_before_price_stops_decreasing": (
            "adverse_turn_timing",
            "frozen OOF adverse-turn head",
            True,
        ),
        "pred_favorable_path_slope_atr_per_hour": (
            "favorable_path_slope",
            "frozen OOF path-slope head",
            True,
        ),
        "catboost_entropy": ("catboost_entropy", "frozen CatBoost entropy", True),
        "base_prediction_uncertainty": (
            "prediction_uncertainty",
            "OOF uncertainty",
            True,
        ),
        "meta_leaf_support_log1p": ("leaf_support", "frozen leaf support", True),
        "base_archetype_label__family__trend": (
            "base_archetype_labels",
            "frozen existing base archetype label",
            True,
        ),
    }
    features.update(
        {
            f"catboost_p_{index}": (
                "catboost_probabilities",
                "frozen CatBoost probability vector",
                True,
            )
            for index in range(len(PATH_SHAPE_TYPES))
        }
    )
    return {
        "schema": runner.HANDOFF_SCHEMA,
        "handoff": {
            "join_mode": "exact_inner_one_to_one",
            "join_keys": list(runner.DEFAULT_ID_COLUMNS),
            "source_artifacts": {
                "alpha": "alpha.parquet",
                "execution": "execution.parquet",
            },
            "row_count": rows,
        },
        "features": {
            name: {
                "family": family,
                "source": source,
                "pre_entry": True,
                "oof_or_frozen": True,
                "available_at_col": "available_at",
                "model_input": model_input,
            }
            for name, (family, source, model_input) in features.items()
        },
    }


def _args(
    input_path: Path, provenance_path: Path, output_dir: Path, **overrides: object
) -> SimpleNamespace:
    values: dict[str, object] = {
        "input": input_path,
        "provenance_json": provenance_path,
        "output_dir": output_dir,
        "id_cols": list(runner.DEFAULT_ID_COLUMNS),
        "timestamp_col": "__ts__",
        "side_col": "side_name",
        "archetype_col": "catboost_archetype",
        "label_end_time_col": "execution_label_end_utc",
        "max_rows": 128,
        "max_span_days": 10.0,
        "n_splits": 2,
        "min_train_rows": 20,
        "hpo_trials": 0,
        "n_estimators": 10,
        "early_stopping_rounds": 3,
        "n_jobs": 1,
        "no_ablations": False,
        "disable_timing_risk_head": True,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_inputs(
    tmp_path: Path, frame: pd.DataFrame | None = None
) -> tuple[Path, Path, pd.DataFrame]:
    frame = _frame() if frame is None else frame
    handoff = tmp_path / "handoff.parquet"
    provenance = tmp_path / "provenance.json"
    frame.to_parquet(handoff, index=False)
    provenance.write_text(json.dumps(_provenance(len(frame))), encoding="utf-8")
    return handoff, provenance, frame


def test_dry_run_requires_exact_join_and_complete_frozen_inputs(tmp_path: Path) -> None:
    handoff, provenance, frame = _write_inputs(tmp_path)
    paths = runner.run(_args(handoff, provenance, tmp_path / "output", dry_run=True))
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["input"]["rows"] == len(frame)
    assert manifest["trainer_config"]["purge_hours"] == 12.0
    assert "OOF/frozen" in manifest["leakage_contract"]


def test_dry_run_uses_provenance_join_keys_when_id_columns_are_omitted(
    tmp_path: Path,
) -> None:
    handoff, provenance, _frame_value = _write_inputs(tmp_path)
    paths = runner.run(
        _args(
            handoff,
            provenance,
            tmp_path / "output",
            dry_run=True,
            id_cols=None,
        )
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["identity_columns"] == list(runner.DEFAULT_ID_COLUMNS)


def test_production_mode_resolves_full_oof_defaults_and_defers_timing(
    tmp_path: Path,
) -> None:
    handoff, provenance, _frame_value = _write_inputs(tmp_path)
    paths = runner.run(
        _args(
            handoff,
            provenance,
            tmp_path / "production",
            dry_run=True,
            production=True,
            timestamp_col=None,
            max_rows=None,
            max_span_days=None,
            n_splits=None,
            min_train_rows=None,
            hpo_trials=None,
            n_estimators=None,
            early_stopping_rounds=None,
            n_jobs=None,
            disable_timing_risk_head=False,
            enable_timing_risk_head=False,
        )
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["run_mode"] == "production"
    assert manifest["trainer_config"]["decision_time_col"] == "execution_decision_utc"
    assert manifest["trainer_config"]["n_splits"] == 3
    assert manifest["trainer_config"]["hpo_trials"] == 40
    assert manifest["trainer_config"]["n_estimators"] == 1_500
    assert manifest["timing_risk_head_enabled"] is False


def test_strict_handoff_rejects_late_probability_and_duplicate_identity(
    tmp_path: Path,
) -> None:
    handoff, provenance, frame = _write_inputs(tmp_path)
    late = frame.copy()
    late.loc[0, "available_at"] = late.loc[0, "__ts__"] + pd.Timedelta(seconds=1)
    late.to_parquet(handoff, index=False)
    with pytest.raises(ValueError, match="available after"):
        runner.run(_args(handoff, provenance, tmp_path / "late", dry_run=True))

    duplicate = frame.copy()
    duplicate.loc[1, list(runner.DEFAULT_ID_COLUMNS)] = duplicate.loc[
        0, list(runner.DEFAULT_ID_COLUMNS)
    ]
    duplicate.to_parquet(handoff, index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        runner.run(_args(handoff, provenance, tmp_path / "duplicate", dry_run=True))


def test_handoff_checks_identity_after_utc_canonicalization(tmp_path: Path) -> None:
    handoff, provenance_path, frame = _write_inputs(tmp_path)
    provenance, payload = runner._load_provenance(provenance_path)
    duplicate = frame.copy()
    duplicate["__ts__"] = duplicate["__ts__"].astype(object)
    duplicate.loc[1, list(runner.DEFAULT_ID_COLUMNS)[1:]] = duplicate.loc[
        0, list(runner.DEFAULT_ID_COLUMNS)[1:]
    ]
    # This is the same instant as row 0, but a raw object comparison would not
    # treat this offset-form string as identical to the UTC Timestamp.
    duplicate.loc[1, "__ts__"] = "2025-12-31T19:00:00-05:00"
    with pytest.raises(ValueError, match="one-to-one"):
        runner._validate_handoff(
            duplicate,
            provenance=provenance,
            provenance_payload=payload,
            id_columns=runner.DEFAULT_ID_COLUMNS,
            timestamp_col="__ts__",
            side_col="side_name",
            archetype_col="catboost_archetype",
            label_end_time_col=None,
            max_span_days=10.0,
        )


def test_runner_persists_oof_bundle_reports_and_top10_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff, provenance_path, frame = _write_inputs(tmp_path)
    expected = frame["execution_net_ev_12h"].to_numpy()
    oof = pd.DataFrame(
        {
            "direct__all_features": expected,
            "residual__all_features": expected * 0.5,
            "direct__alpha_only": expected * 0.25,
        }
    )
    bundle = SimpleNamespace(
        oof_predictions=oof,
        oof_provenance=pd.DataFrame(
            {
                "execution_ev_oof_fold": pd.array([0] * len(frame), dtype="Int64"),
                "execution_ev_oof_validation_start_utc": frame["__ts__"],
                "execution_ev_oof_train_decision_cutoff_utc": frame["__ts__"],
            }
        ),
        report={
            "oof_contract": "outer expanding purged folds; fold HPO/early-stop/calibration are training-only",
            "folds": [{"fold": 0}],
            "diagnostics": pd.DataFrame({"scope": ["overall"], "rows": [len(frame)]}),
        },
    )
    monkeypatch.setattr(
        runner, "train_execution_ev_meta", lambda *args, **kwargs: bundle
    )

    def fake_save(_bundle: object, path: Path) -> Path:
        path = Path(path)
        path.write_bytes(b"bundle")
        return path

    monkeypatch.setattr(runner, "save_execution_ev_bundle", fake_save)

    def fake_report(_bundle: object, output_dir: Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        diagnostics = output_dir / "execution_ev_diagnostics.csv"
        report = output_dir / "execution_ev_report.json"
        diagnostics.write_text("scope,rows\noverall,96\n", encoding="utf-8")
        report.write_text("{}\n", encoding="utf-8")
        return {"diagnostics": diagnostics, "report": report}

    monkeypatch.setattr(runner, "write_execution_ev_report", fake_report)
    paths = runner.run(_args(handoff, provenance_path, tmp_path / "trained"))
    assert all(path.exists() for path in paths.values())
    winner = json.loads(paths["winner"].read_text())
    assert winner["winner"]["prediction"] == "direct__all_features"
    assert winner["selection_scope"] == "direct_vs_residual_all_features_only"
    ledger = pd.read_parquet(paths["oof"])
    assert {
        "direct__all_features",
        "direct__all_features__is_oof",
        "execution_ev_oof_fold",
        "execution_ev_oof_train_decision_cutoff_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_label_end_utc",
    }.issubset(ledger.columns)
    assert ledger["direct__all_features__is_oof"].all()


def test_runner_never_promotes_ablation_over_full_input_model_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff, provenance_path, frame = _write_inputs(tmp_path)
    expected = frame["execution_net_ev_12h"].to_numpy()
    bundle = SimpleNamespace(
        oof_predictions=pd.DataFrame(
            {
                "direct__all_features": expected * 0.50,
                "residual__all_features": expected * 0.75,
                "direct__without_catboost": expected,
            }
        ),
        oof_provenance=None,
        report={"oof_contract": "test", "folds": [], "diagnostics": pd.DataFrame()},
    )
    monkeypatch.setattr(
        runner, "train_execution_ev_meta", lambda *args, **kwargs: bundle
    )

    def fake_save(_bundle: object, path: Path) -> Path:
        Path(path).write_bytes(b"bundle")
        return Path(path)

    def fake_report(_bundle: object, output_dir: Path) -> dict[str, Path]:
        report = Path(output_dir) / "execution_ev_report.json"
        report.write_text("{}\n", encoding="utf-8")
        return {"report": report}

    monkeypatch.setattr(runner, "save_execution_ev_bundle", fake_save)
    monkeypatch.setattr(runner, "write_execution_ev_report", fake_report)
    paths = runner.run(_args(handoff, provenance_path, tmp_path / "winner-scope"))
    payload = json.loads(paths["winner"].read_text())
    assert payload["winner"]["prediction"] == "residual__all_features"
    assert payload["best_diagnostic_arm"]["prediction"] == "direct__without_catboost"


def test_runner_smoke_trains_and_persists_real_bundle(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    handoff, provenance, _ = _write_inputs(tmp_path)
    paths = runner.run(
        _args(
            handoff,
            provenance,
            tmp_path / "real-smoke",
            no_ablations=True,
            n_estimators=8,
            early_stopping_rounds=3,
        )
    )
    assert paths["bundle"].is_file()
    leaderboard = pd.read_csv(paths["leaderboard"])
    assert {"baseline", "direct", "residual"} == set(leaderboard["mode"])
    assert leaderboard["top10_mean_net_ev"].notna().all()


def test_runner_smoke_trains_and_persists_timing_risk_companion(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    frame = _frame(rows=192)
    favorable = np.arange(len(frame)) % 3 != 0
    frame["execution_net_ev_12h"] = np.where(favorable, 0.012, -0.008)
    frame["execution_gross_ev_12h"] = frame["execution_net_ev_12h"] + 0.003
    frame["execution_exit_hour"] = np.where(favorable, 2 + np.arange(len(frame)) % 7, 1)
    frame["execution_exit_reason"] = np.where(favorable, "trailing", "full_stop")
    handoff, provenance, _ = _write_inputs(tmp_path, frame)
    paths = runner.run(
        _args(
            handoff,
            provenance,
            tmp_path / "timing-risk-smoke",
            max_rows=256,
            no_ablations=True,
            disable_timing_risk_head=False,
            enable_timing_risk_head=True,
            n_estimators=8,
            early_stopping_rounds=3,
        )
    )
    assert paths["timing_risk_bundle"].is_file()
    diagnostics = pd.read_csv(paths["timing_risk_diagnostics"])
    assert {"overall", "side", "month"}.issubset(diagnostics["scope"])
    assert (
        diagnostics.loc[diagnostics["scope"] == "overall", "loss_brier"].notna().all()
    )


def test_runner_rejects_input_before_loading_when_row_smoke_cap_is_exceeded(
    tmp_path: Path,
) -> None:
    handoff, provenance, _ = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="before loading"):
        runner.run(
            _args(handoff, provenance, tmp_path / "too-large", max_rows=4, dry_run=True)
        )
