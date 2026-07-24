from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_base_side_target_ablation_matrix import (
    MatrixConfig,
    build_arm_commands,
    build_stage_c_command,
    choose_b_scope,
    report_matrix,
    stage_c_contract,
)


def _config(tmp_path: Path, *, frozen_state: Path | None = None) -> MatrixConfig:
    params = tmp_path / "l2.json"
    params.write_text(
        json.dumps(
            {
                "params": {
                    "target_mode": "target_soft",
                    "weight_arm": "W7_timestamp_balanced",
                    "loss_function": "regression",
                    "n_estimators": 147,
                    "learning_rate": 0.02,
                    "num_leaves": 15,
                    "max_depth": 4,
                    "min_child_samples": 98,
                    "subsample": 0.8,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.34,
                    "min_split_gain": 0.0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return MatrixConfig(
        labels_path=tmp_path / "labels",
        feature_dir=tmp_path / "features",
        feature_list_csv=tmp_path / "features.csv",
        output_dir=tmp_path / "out",
        fixed_params_json=params,
        frozen_ae_gmm_state=frozen_state,
    )


def _arg_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_fixed_window_commands_keep_base_l2_contract_and_no_growing_windows(tmp_path: Path) -> None:
    commands = build_arm_commands(_config(tmp_path))

    a0 = commands["A0_shared_l2"]
    a = commands["A_per_side_l2"]
    b = commands["B_corrfirst_if_per_side"]

    for command in (a0, a, b):
        assert "--single-fit-oos-window" in command
        assert _arg_after(command, "--train-window-days") == "365"
        assert _arg_after(command, "--label-path-purge-hours") == "25.0"
        assert _arg_after(command, "--months") == "2026-04,2026-05,2026-06"
        assert _arg_after(command, "--fixed-params-json").endswith("l2.json")
        assert "--rerun-hpo" not in command
        assert "--no-save-final-model" in command

    assert _arg_after(a0, "--model-side-scope") == "shared"
    assert _arg_after(a0, "--feature-selection-method") == "mda"
    assert "--refit-cycle-ae-gmm" in a0

    assert _arg_after(a, "--model-side-scope") == "per_side"
    assert _arg_after(a, "--feature-selection-method") == "archetype_prescreen_side_mda"
    assert _arg_after(a, "--fixed-ae-gmm-state-pkl").endswith(
        "A0_shared_l2/_feature_selection_phase/ae_gmm_states/cycle__global_state.pkl"
    )

    assert _arg_after(b, "--model-side-scope") == "per_side"
    assert _arg_after(b, "--feature-selection-method") == "archetype_prescreen_side_mda_corrfirst"


def test_provided_aegmm_state_is_reused_by_every_executed_arm(tmp_path: Path) -> None:
    frozen = tmp_path / "cycle_state.pkl"
    frozen.write_bytes(b"state")
    commands = build_arm_commands(_config(tmp_path, frozen_state=frozen))

    for command in commands.values():
        assert _arg_after(command, "--fixed-ae-gmm-state-pkl") == str(frozen)
        assert "--refit-cycle-ae-gmm" not in command


def test_conditional_b_uses_per_side_only_for_strict_a_win() -> None:
    assert choose_b_scope(0.0100, 0.0101) == "per_side"
    assert choose_b_scope(0.0100, 0.0100) == "shared"
    assert choose_b_scope(0.0100, 0.0099) == "shared"
    assert choose_b_scope(float("nan"), 0.0200) == "shared"


def test_matrix_rejects_meta_residual_parameter_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.fixed_params_json.write_text(
        json.dumps({"params": {"target_mode": "residual_net_ev_after_1pct", "loss_function": "regression"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="corrected base soft target"):
        build_arm_commands(config)


def test_stage_c_command_describes_executable_hierarchical_contract(tmp_path: Path) -> None:
    command = build_stage_c_command(
        _config(tmp_path),
        source_arm_name="B_corrfirst_pack_b",
        model_side_scope="per_side",
    )
    spec = stage_c_contract(source_arm_name="B_corrfirst_pack_b", model_side_scope="per_side")

    assert command[1].endswith("run_base_side_target_geometry_hpo.py")
    assert _arg_after(command, "--model-side-scope") == "per_side"
    assert _arg_after(command, "--source-arm-dir").endswith("B_corrfirst_pack_b")
    assert spec["target"].startswith("continuous corrected base")
    assert spec["weight_search"]["strength_ratio_continuous"] == [3.0, 12.0]
    assert "full LightGBM finalists" in spec["selection"]


def test_report_intersects_identical_rows_before_comparing_arms(tmp_path: Path) -> None:
    timestamps = [
        "2026-04-01T00:00:00Z",
        "2026-04-01T00:00:00Z",
        "2026-04-01T01:00:00Z",
        "2026-04-01T01:00:00Z",
    ]
    source = {
        "__ts__": timestamps,
        "__symbol__": ["AAA", "BBB", "AAA", "BBB"],
        "side_name": ["long", "long", "short", "short"],
        "__first_touch_net__": [0.01, -0.01, 0.02, -0.02],
        "__first_touch_round_trip_cost__": [0.01] * 4,
        "__archetype_label_family__": ["trend"] * 4,
    }
    a0 = pd.DataFrame({**source, "score": [0.9, 0.2, 0.8, 0.1]})
    a = pd.DataFrame({**source, "score": [0.2, 0.9, 0.1, 0.8]})
    # Extra A-only row must be excluded from every arm's reported denominator.
    a = pd.concat(
        [
            a,
            pd.DataFrame(
                {
                    "__ts__": ["2026-04-01T02:00:00Z"],
                    "__symbol__": ["EXTRA"],
                    "side_name": ["long"],
                    "__first_touch_net__": [0.50],
                    "__first_touch_round_trip_cost__": [0.01],
                    "__archetype_label_family__": ["trend"],
                    "score": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    a0_path = tmp_path / "a0.parquet"
    a_path = tmp_path / "a.parquet"
    a0.to_parquet(a0_path, index=False)
    a.to_parquet(a_path, index=False)

    paths = report_matrix(
        {"A0_shared_l2": a0_path, "A_per_side_l2": a_path},
        output_dir=tmp_path / "report",
        oos_months=("2026-04", "2026-05", "2026-06"),
    )

    coverage = pd.read_csv(paths["coverage"])
    assert set(coverage["common_rows"]) == {4}
    metrics = pd.read_csv(paths["metrics"])
    overall_top10 = metrics.loc[
        metrics["scope"].eq("overall") & (metrics["top_frac"] == 0.10)
    ]
    assert set(overall_top10["candidate_rows"]) == {4}


def test_report_rebases_old_cost_to_exact_symbol_p90_spread_plus_fee(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-01T00:00:00Z"], utc=True
            ),
            "__symbol__": ["AAA/USD:USD", "BBB/USD:USD"],
            "side_name": ["long", "long"],
            "__first_touch_net__": [0.01, 0.00],
            "__first_touch_round_trip_cost__": [0.01, 0.01],
            "score": [0.9, 0.1],
        }
    )
    ledger = tmp_path / "ledger.parquet"
    source.to_parquet(ledger, index=False)
    spread = pd.DataFrame(
        {
            "observed_ts": pd.to_datetime(
                [
                    "2026-06-01T00:00:00Z",
                    "2026-06-02T00:00:00Z",
                    "2026-06-01T00:00:00Z",
                    "2026-06-02T00:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAA/USD:USD", "AAA/USD:USD", "BBB/USD:USD", "BBB/USD:USD"],
            "spread_bps": [10.0, 20.0, 30.0, 40.0],
        }
    )
    spread_path = tmp_path / "spread.parquet"
    spread.to_parquet(spread_path, index=False)

    paths = report_matrix(
        {"A0_shared_l2": ledger},
        output_dir=tmp_path / "cost_report",
        oos_months=("2026-04", "2026-05", "2026-06"),
        spread_snapshot_path=spread_path,
        spread_quantile=0.90,
        fee_round_trip_pct=0.0015,
    )

    metrics = pd.read_csv(paths["metrics"])
    top10 = metrics.loc[
        metrics["scope"].eq("overall") & metrics["top_frac"].eq(0.10)
    ].iloc[0]
    # AAA is selected: gross 2%; p90 spread 19 bps; fee 15 bps.
    assert top10["mean_gross_ev"] == pytest.approx(0.02)
    assert top10["mean_net_ev"] == pytest.approx(0.02 - 0.0019 - 0.0015)
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["spread_cost_rebase"]["required_symbols"] == 2
    assert provenance["spread_cost_rebase"]["missing_symbols"] == []


def test_report_cost_rebase_rejects_missing_symbol_spread(tmp_path: Path) -> None:
    ledger = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
            "__symbol__": ["MISSING/USD:USD"],
            "side_name": ["long"],
            "__first_touch_net__": [0.01],
            "__first_touch_round_trip_cost__": [0.01],
            "score": [0.9],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    spread_path = tmp_path / "spread.parquet"
    pd.DataFrame(
        {
            "observed_ts": pd.to_datetime(["2026-06-01T00:00:00Z"], utc=True),
            "symbol": ["OTHER/USD:USD"],
            "spread_bps": [10.0],
        }
    ).to_parquet(spread_path, index=False)

    with pytest.raises(ValueError, match="exact symbol coverage"):
        report_matrix(
            {"A0_shared_l2": ledger_path},
            output_dir=tmp_path / "report",
            oos_months=("2026-04", "2026-05", "2026-06"),
            spread_snapshot_path=spread_path,
        )
