from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_vi_archetype_runner import (
    AW_CONTRACTS,
    CAUSAL_COMPONENTS,
    PATH_COMPONENTS,
    StageVIArmTrainingResult,
    StageVIRunnerError,
    StageVIRunnerSpec,
    materialize_stage_vi_view_contract,
    run_stage_vi_archetype_funnel,
    stage_vi_candidate_grid,
    stage_vi_sequential_funnel_grid,
)


def _ledger(rows: int = 480) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    decision = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    mode = np.arange(rows) % 3
    net = 20.0 + 15.0 * mode + rng.normal(0.0, 3.0, rows)
    setup = mode + rng.normal(0.0, 0.1, rows)
    return pd.DataFrame({
        "candidate_id": [f"candidate-{index:05d}" for index in range(rows)],
        "symbol": np.where(np.arange(rows) % 3, "BTC", "ETH"),
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "side_name": np.where(np.arange(rows) % 2, "long", "short"),
        "exact_net_bps": net,
        "exact_gross_bps": net + 100.0,
        "atr_setup": setup,
        "trend_signal": setup * 0.7,
        "r3_p_clear": np.clip(0.4 + setup / 10.0, 0.0, 1.0),
        "base_r3_entropy": 0.8 - setup / 20.0,
        "market_regime": mode + rng.normal(0.0, 0.2, rows),
        "liquidity_context": rng.normal(size=rows),
        "event_upper": (mode == 2).astype(float),
        "event_lower": (mode == 0).astype(float),
        "event_timeout": (mode == 1).astype(float),
        "time_to_first_touch": 1.0 + mode,
        "same_bar_conflict": np.zeros(rows),
        "mfe": 0.5 + mode,
        "mae": -0.2 - mode / 2.0,
        "mae_before_mfe": -0.1 - mode / 3.0,
        "time_to_mfe": 2.0 + mode,
        "time_to_mae": 1.0 + mode,
        "terminal_peak_ratio": np.clip(0.9 - mode / 10.0, 0.0, 1.0),
        "post_clear_change": mode / 10.0,
        "giveback_fraction": mode / 5.0,
        "retained_positive_gross": 1.0 - mode / 10.0,
        "retained_positive_net": 0.8 - mode / 10.0,
        "path_efficiency": 0.5 + mode / 10.0,
        "directional_consistency": 0.6 + mode / 10.0,
        "future_slope": mode / 20.0,
        "future_slope_r2": 0.5 + mode / 10.0,
        "reversal_count": mode.astype(float),
        "jump_concentration": 0.2 + mode / 10.0,
        "future_volatility": 0.4 + mode / 10.0,
        "path_certainty": 0.6 + mode / 10.0,
        "economic_bucket": mode.astype(str),
        "control_score": np.linspace(0.0, 1.0, rows),
        "base_archetype_score": np.linspace(0.0, 1.0, rows),
        "meta_archetype_score": np.linspace(0.0, 1.0, rows),
        "both_archetype_score": np.linspace(0.0, 1.0, rows),
        "control_is_strict_oof": True,
        "base_is_strict_oof": True,
        "meta_is_strict_oof": True,
        "both_is_strict_oof": True,
    })


def _views(frame: pd.DataFrame):
    selected = (
        "atr_setup", "trend_signal", "r3_p_clear", "base_r3_entropy",
        "market_regime", "liquidity_context",
    )
    config = {
        "base_shared_feature_keys": ["BASE_COMPACT"],
        "BASE_COMPACT": list(selected),
        "base_long_feature_keys": [], "base_short_feature_keys": [],
        "meta_shared_feature_keys": [], "meta_product_feature_keys": [],
    }
    return materialize_stage_vi_view_contract(
        frame, config=config, selected_causal_columns=selected,
    )


def _single_candidate_spec(candidate_id: str) -> StageVIRunnerSpec:
    return StageVIRunnerSpec(
        folds=3,
        min_side_rows=40,
        min_component_rows=5,
        arm_score_columns_by_candidate={
            candidate_id: {
                "control": "control_score",
                "base": "base_archetype_score",
                "meta": "meta_archetype_score",
                "both": "both_archetype_score",
            }
        },
        arm_oof_flag_columns_by_candidate={
            candidate_id: {
                "control": "control_is_strict_oof",
                "base": "base_is_strict_oof",
                "meta": "meta_is_strict_oof",
                "both": "both_is_strict_oof",
            }
        },
    )


def test_view_contract_materializes_cf0_cf4_pf0_pf4_from_config_and_ledger() -> None:
    frame = _ledger()
    views = _views(frame)
    assert set(views.causal_views) == {"CF0", "CF1", "CF2", "CF3", "CF4"}
    assert set(views.path_views) == {"PF0", "PF1", "PF2", "PF3", "PF4"}
    assert set(views.causal_views["CF0"]).issubset(
        {"atr_setup", "trend_signal", "r3_p_clear", "base_r3_entropy", "market_regime", "liquidity_context"}
    )
    assert views.multiview_sources.keys() == {"setup", "base_trust", "regime"}


def test_grid_enforces_distinct_k_and_all_aw_contracts() -> None:
    grid = stage_vi_candidate_grid(_views(_ledger()), StageVIRunnerSpec())
    causal = [config for _name, config in grid if config.view.kind == "causal"]
    path = [config for _name, config in grid if config.view.kind == "path"]
    assert {config.components for config in causal} == set(CAUSAL_COMPONENTS)
    assert {config.components for config in path} == set(PATH_COMPONENTS)
    assert set(AW_CONTRACTS) == {"AW0", "AW1", "AW2", "AW3", "AW4", "AW5"}
    assert all(config.side_col == "side_name" for config in grid[0][1:])
    with pytest.raises(StageVIRunnerError, match=r"K=\{3,4,5,6\}"):
        StageVIRunnerSpec(causal_components=(3, 4, 5)).validate()


def test_default_sequential_funnel_is_bounded_and_full_grid_is_explicit() -> None:
    views = _views(_ledger())
    full = stage_vi_candidate_grid(views)
    bounded = stage_vi_sequential_funnel_grid(views)
    assert len(full) == 925
    assert len(bounded) == 31
    assert {config.method for _candidate, config in bounded} == {
        "kmeans", "gmm_diag", "gmm_full", "gmm_pca_diag", "ae_gmm_diag",
    }
    assert len(bounded) < len(full) / 20


def test_runner_publishes_strict_oof_matched_controls_and_decision_matrix(tmp_path: Path) -> None:
    frame = _ledger()
    views = _views(frame)
    candidate = "CF1__kmeans__k3__AW0"
    output = tmp_path / "stage_vi_bundle"
    result = run_stage_vi_archetype_funnel(
        frame,
        views=views,
        output_directory=output,
        spec=_single_candidate_spec(candidate),
        candidate_ids=[candidate],
    )
    assert result.candidate_audit.candidate_id.tolist() == [candidate]
    assert set(result.comparison.arm) == {"control", "base", "meta", "both"}
    assert result.comparison.global_ranking.all()
    assert not result.decision_matrix.empty
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["strict_oof_causal_recognisers"] is True
    assert manifest["positive_label_only_discovery"] is True
    assert manifest["side_local_discovery"] is True
    assert manifest["aw_contracts"]["AW3"] == "mandatory_side_local_fit"
    assert (output / "checksums.json").is_file()
    feature_path = next((output / "candidates").glob("*/strict_oof_features.parquet"))
    features = pd.read_parquet(feature_path)
    assert {"candidate_id", "decision_ts", "side_name"}.issubset(features)
    with pytest.raises(StageVIRunnerError, match="already exists"):
        run_stage_vi_archetype_funnel(
            frame,
            views=views,
            output_directory=output,
            spec=_single_candidate_spec(candidate),
            candidate_ids=[candidate],
        )


def test_runner_rejects_non_oof_matched_control(tmp_path: Path) -> None:
    frame = _ledger()
    frame.loc[0, "meta_is_strict_oof"] = False
    candidate = "CF1__kmeans__k3__AW0"
    with pytest.raises(StageVIRunnerError, match="meta scores must be strict OOF"):
        run_stage_vi_archetype_funnel(
            frame,
            views=_views(frame),
            output_directory=tmp_path / "rejected",
            spec=_single_candidate_spec(candidate),
            candidate_ids=[candidate],
        )


def test_runner_rejects_generic_scores_for_single_candidate(tmp_path: Path) -> None:
    frame = _ledger()
    with pytest.raises(StageVIRunnerError, match="every candidate requires"):
        run_stage_vi_archetype_funnel(
            frame,
            views=_views(frame),
            output_directory=tmp_path / "rejected_single_generic",
            spec=StageVIRunnerSpec(folds=3, min_side_rows=40, min_component_rows=5),
            candidate_ids=["CF1__kmeans__k3__AW0"],
        )


def test_runner_rejects_generic_scores_for_multiple_candidates(tmp_path: Path) -> None:
    frame = _ledger()
    with pytest.raises(StageVIRunnerError, match="false attribution"):
        run_stage_vi_archetype_funnel(
            frame,
            views=_views(frame),
            output_directory=tmp_path / "rejected_generic",
            spec=StageVIRunnerSpec(folds=3, min_side_rows=40, min_component_rows=5),
            candidate_ids=[
                "CF1__kmeans__k3__AW0",
                "CF2__kmeans__k3__AW0",
            ],
        )


def test_candidate_features_drive_distinct_arm_fits_and_hash_bindings(
    tmp_path: Path,
) -> None:
    frame = _ledger()
    seen: dict[str, str] = {}

    def trainer(request):
        seen[request.candidate_id] = request.feature_sha256
        probability_columns = [
            column for column in request.archetype_feature_columns
            if "prob__" in column and not column.endswith("unknown")
        ]
        signal = request.archetype_features[probability_columns[0]].to_numpy(float)
        feature_offset = int(request.feature_sha256[:8], 16) / float(16**8)
        control = np.linspace(0.0, 1.0, len(signal))
        flags = {arm: np.ones(len(signal), dtype=bool) for arm in (
            "control", "base", "meta", "both"
        )}
        return StageVIArmTrainingResult(
            candidate_ids=request.candidate_ids.copy(),
            scores={
                "control": control,
                "base": signal + feature_offset,
                "meta": -signal + feature_offset,
                "both": 2.0 * signal + feature_offset,
            },
            oof_flags=flags,
            provenance={
                "strict_oof": True,
                "candidate_feature_sha256": request.feature_sha256,
                "archetype_feature_columns": list(
                    request.archetype_feature_columns
                ),
                "arm_feature_usage": {
                    "control": "none", "base": "base", "meta": "meta",
                    "both": "base_and_meta",
                },
            },
        )

    candidates = ["CF1__kmeans__k3__AW0", "CF2__kmeans__k3__AW0"]
    output = tmp_path / "trained"
    run_stage_vi_archetype_funnel(
        frame,
        views=_views(frame),
        output_directory=output,
        spec=StageVIRunnerSpec(folds=3, min_side_rows=40, min_component_rows=5),
        candidate_ids=candidates,
        arm_trainer=trainer,
    )
    bindings = json.loads((output / "score_bindings.json").read_text())
    assert set(seen) == set(candidates)
    assert len(set(seen.values())) == 2
    assert {row["score_source"] for row in bindings} == {"arm_trainer"}
    assert len({row["score_sha256"] for row in bindings}) == 2
    assert all(
        row["candidate_feature_sha256"] == seen[row["candidate_id"]]
        == row["provenance"]["candidate_feature_sha256"]
        for row in bindings
    )
    assert all(
        row["archetype_feature_columns"]
        == row["provenance"]["archetype_feature_columns"]
        for row in bindings
    )
