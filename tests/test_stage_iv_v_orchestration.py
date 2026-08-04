from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iv_broad_to_tail import StageIVPlan
from extreme_price_movements.stage_iv_v_orchestration import (
    StageIVCell,
    StageIVCellSpec,
    StageVControllerArm,
    StageIVVOrchestrationError,
    apply_stage_v_contract_to_stage_iv_plan,
    freeze_stage_v_feature_contract,
    run_stage_iv_sequential_sweep,
    select_stage_v_controller,
)
from extreme_price_movements.stage_v_drift_ood import (
    STAGE_V_FEATURE_COLUMNS,
    StageVContract,
    fit_stage_v_drift_ood_state,
)


class _FirstColumnModel:
    def __init__(self, column: str, mean: float) -> None:
        self.column = column
        self.mean = mean

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(np.float32) + np.float32(self.mean * 0.0001)


def _fitter(X, y, _weight, _layer, _params):
    raw = [name for name in X.columns if not name.startswith("__stage_iv_")]
    return _FirstColumnModel(raw[0], float(np.mean(y)))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _plan(side: str, *, fraction: float, broad_burn: int = 24, tail_burn: int = 12, meta_burn: int = 10, route: str = "both") -> StageIVPlan:
    timestamps = 180
    per_timestamp = 2
    n = timestamps * per_timestamp
    rng = np.random.default_rng(42 if side == "long" else 8)
    ts = pd.date_range("2024-01-01", periods=timestamps, freq="h", tz="UTC").repeat(per_timestamp)
    rank = np.tile(np.asarray([0.2, 0.8], dtype=np.float32), timestamps)
    frame = pd.DataFrame({
        "broad": rank + rng.normal(0.0, 0.01, n),
        "tail": rank + rng.normal(0.0, 0.01, n),
        "meta": rank + rng.normal(0.0, 0.01, n),
        **{column: rng.normal(size=n).astype("float32") for column in STAGE_V_FEATURE_COLUMNS},
    }).astype("float32")
    return StageIVPlan(
        side=side,
        candidate_ids=[f"{side}-{i}" for i in range(n)],
        frame=frame,
        base_target=(rank * 100.0 + rng.normal(0.0, 1.0, n)).astype("float32"),
        exact_net_bps=(rank * 200.0 - 100.0 + rng.normal(0.0, 1.0, n)).astype("float32"),
        decision_timestamps=ts,
        label_available_timestamps=ts + pd.Timedelta(hours=13),
        broad_feature_names=("broad",), tail_feature_names=("tail",), meta_feature_names=("meta",),
        tail_fraction=fraction,
        broad_min_train_rows=broad_burn,
        tail_min_train_rows=tail_burn,
        meta_min_train_rows=meta_burn,
        min_handoff_history_rows=12,
        n_validation_folds=3,
        broad_output_route=route,
        burn_in_months=0,
        # Compact fixture only: production retains a 100-row prior-resolved
        # value-map floor.
        value_map_min_history_rows=8,
        value_map_bins=4,
    )


def _cell(cell_id: str, fraction: float, *, burns: tuple[int, int, int], route: str) -> StageIVCell:
    broad, tail, meta = burns
    return StageIVCell(
        spec=StageIVCellSpec(
            cell_id=cell_id,
            tail_fraction=fraction,
            broad_min_train_rows=broad,
            tail_min_train_rows=tail,
            meta_min_train_rows=meta,
            min_handoff_history_rows=12,
            broad_output_route=route,
        ),
        plans=(
            _plan("long", fraction=fraction, broad_burn=broad, tail_burn=tail, meta_burn=meta, route=route),
            _plan("short", fraction=fraction, broad_burn=broad, tail_burn=tail, meta_burn=meta, route=route),
        ),
        source_lineage={"dataset": _digest("same-candidate-dataset"), "labels": _digest("same-label-contract")},
    )


def _mda_audit() -> pd.DataFrame:
    return pd.DataFrame({
        "group_id": ["g0", "g1"], "group_kind": ["correlation", "correlation"],
        "features": ["a|b", "c|d"], "group_mda_lower_95": [0.1, 0.05],
    })


def _reference(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(100)
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a + rng.normal(0.0, 0.1, n), "c": rng.normal(size=n), "d": rng.normal(size=n)})


def test_stage_iv_primary_sweep_is_explicit_sequential_matched_and_freezes_compact_winner() -> None:
    cells = (
        _cell("x20", 0.20, burns=(24, 12, 10), route="both"),
        _cell("x30", 0.30, burns=(28, 14, 11), route="tail"),
        _cell("x40", 0.40, burns=(26, 13, 12), route="meta"),
        _cell("x50", 0.50, burns=(30, 15, 13), route="neither"),
    )
    result = run_stage_iv_sequential_sweep(cells, fitter=_fitter)
    pooled = result.matched_metrics.loc[result.matched_metrics.scope.eq("pooled_global")]
    assert set(pooled.cell_id) == {"x20", "x30", "x40", "x50"}
    assert pooled.matched_population_sha256.nunique() == 1
    assert pooled.comparison_population.eq(
        "intersection_of_final_strict_oof_rows_before_one_pooled_global_rank"
    ).all()
    assert result.manifest["execution"] == "sequential_explicit_cells_no_factorial_expansion"
    assert [row["cell_id"] for row in result.manifest["cells"]] == ["x20", "x30", "x40", "x50"]
    assert result.winner.cell_id in {"x20", "x30", "x40", "x50"}
    assert result.manifest["winner"]["cell_input_sha256"] == result.winner.cell_input_sha256
    assert result.manifest["compact_outputs"] == ("matched_metrics", "winner", "cell_input_hashes")


def test_stage_iv_primary_sweep_rejects_partial_x_grid_and_plan_spec_drift() -> None:
    with pytest.raises(StageIVVOrchestrationError, match="x=20/30/40/50%"):
        run_stage_iv_sequential_sweep((_cell("x20", 0.20, burns=(24, 12, 10), route="both"),), fitter=_fitter)
    bad = _cell("x20", 0.20, burns=(24, 12, 10), route="both")
    bad_plan = _plan("long", fraction=0.20, broad_burn=99, tail_burn=12, meta_burn=10, route="both")
    bad = StageIVCell(bad.spec, (bad_plan,), bad.source_lineage)
    with pytest.raises(StageIVVOrchestrationError, match="does not match"):
        bad.validate()


def test_stage_v_freeze_is_side_layer_isolated_and_stage_iv_uses_inputs_not_reranking() -> None:
    reference = _reference()
    base_state = fit_stage_v_drift_ood_state(
        reference, contract=StageVContract("long", "base"), mda_audit=_mda_audit()
    )
    meta_state = fit_stage_v_drift_ood_state(
        reference, contract=StageVContract("long", "meta"), mda_audit=_mda_audit()
    )
    base = freeze_stage_v_feature_contract(
        contract=StageVContract("long", "base"), raw_feature_names=("broad", "tail"),
        state=base_state, controller="soft_ood",
    )
    meta = freeze_stage_v_feature_contract(
        contract=StageVContract("long", "meta"), raw_feature_names=("meta",),
        state=meta_state, controller="grouped_ood",
    )
    assert base.model_feature_contract_sha256 == hashlib.sha256(
        b'["broad","tail","stage_v_reference_ready","stage_v_ood_score"]'
    ).hexdigest()
    updated = apply_stage_v_contract_to_stage_iv_plan(
        _plan("long", fraction=0.30), base_contract=base, meta_contract=meta
    )
    assert updated.broad_feature_names[-2:] == ("stage_v_reference_ready", "stage_v_ood_score")
    assert updated.tail_feature_names[-2:] == ("stage_v_reference_ready", "stage_v_ood_score")
    assert "stage_v_ood_score" in updated.meta_feature_names
    assert "stage_v_group_drift_max" in updated.meta_feature_names
    assert "stage_v_group_drift_max" not in updated.broad_feature_names
    with pytest.raises(StageIVVOrchestrationError, match="cannot enter"):
        apply_stage_v_contract_to_stage_iv_plan(_plan("short", fraction=0.30), base_contract=base)
    with pytest.raises(StageIVVOrchestrationError, match="duplicates"):
        freeze_stage_v_feature_contract(
            contract=StageVContract("long", "base"),
            raw_feature_names=("broad", "broad"), state=base_state,
            controller="soft_ood",
        )


def test_stage_v_controller_selection_is_matched_and_never_side_or_time_reranked() -> None:
    common = {
        "candidate_key": ["long::a", "short::b", "long::c", "short::d"],
        "side_name": ["long", "short", "long", "short"],
        "decision_ts": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
            "2024-02-01T00:00:00Z", "2024-02-01T00:00:00Z",
        ]),
        "exact_net_bps": [100.0, -100.0, -100.0, -100.0],
        "cost_bps": [100.0] * 4,
    }
    left = pd.DataFrame({**common, "score": [4.0, 3.0, 2.0, 1.0]})
    right = pd.DataFrame({**common, "score": [1.0, 4.0, 3.0, 2.0]})
    reference = _reference()
    def arm(arm_id: str, controller: str) -> StageVControllerArm:
        contracts = []
        for side in ("long", "short"):
            state = fit_stage_v_drift_ood_state(
                reference, contract=StageVContract(side, "base"), mda_audit=_mda_audit()
            )
            contracts.append(freeze_stage_v_feature_contract(
                contract=StageVContract(side, "base"), raw_feature_names=("a",),
                state=state, controller=controller,
            ))
        return StageVControllerArm(arm_id, tuple(contracts))
    arms = {"context": arm("context", "soft_ood"), "control": arm("control", "none")}
    metrics, selected = select_stage_v_controller(
        {"context": left, "control": right}, controller_arms=arms,
        score_column="score", selection_top_fraction=0.25,
    )
    pooled = metrics.loc[metrics.scope.eq("pooled_global")]
    assert selected.winner_arm_id == "context"
    assert pooled.matched_population_sha256.nunique() == 1
    assert pooled.selection.eq("pooled_global_once_no_timestamp_month_or_side_rerank").all()
    assert selected.manifest["ranking"].startswith("pooled_global_once")
    assert set(selected.manifest["controller_contracts"]) == {"context", "control"}
