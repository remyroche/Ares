from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_v_drift_ood_experiment import (
    STAGE_V_EXPERIMENT_SCHEMA,
    StageVExperimentConfig,
    StageVExperimentError,
    StageVLayerSource,
    run_stage_v_drift_ood_ablation,
    _winner,
)


_RAW = ("f_a", "f_b", "f_c", "f_d")


def _audit() -> pd.DataFrame:
    return pd.DataFrame({
        "group_id": ["g_ab", "g_cd"],
        "group_kind": ["correlation", "correlation"],
        "features": ["f_a|f_b", "f_c|f_d"],
        "group_mda_lower_95": [0.15, 0.08],
    })


def _no_positive_audit() -> pd.DataFrame:
    audit = _audit()
    audit["group_mda_lower_95"] = -0.01
    return audit


def _panel(*, layer: str, side: str, start: str, n: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((side, prefix))) % (2**32))
    decision = pd.date_range(start, periods=n, freq="h", tz="UTC")
    f_a = rng.normal(size=n)
    f_b = f_a + rng.normal(scale=0.15, size=n)
    f_c = rng.normal(size=n)
    f_d = f_c + rng.normal(scale=0.20, size=n)
    # A deliberately common-bps, finite target.  The test checks contract and
    # chronology rather than trying to manufacture a profitable strategy.
    net = 65.0 * f_a - 18.0 * f_c + rng.normal(scale=8.0, size=n)
    target = net + (4.0 if layer == "meta" else 0.0)
    return pd.DataFrame({
        # Base and meta must consume the identical same-side candidate rows;
        # only their selected raw contracts differ.
        "candidate_id": [f"{side}-{prefix}-{i}" for i in range(n)],
        "symbol": "BTCUSDT" if side == "long" else "ETHUSDT",
        "side_name": side,
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "exact_net_bps": net,
        "exact_gross_bps": net + 100.0,
        "r3_class": np.select([net < -20.0, net > 20.0], [0, 2], default=1).astype(np.int8),
        f"{layer}_target": target,
        "f_a": f_a,
        "f_b": f_b,
        "f_c": f_c,
        "f_d": f_d,
    })


def _sources(*, mutate_oos_target: bool = False, positive_mda: bool = True, collide_oos_with_oof: bool = False) -> list[StageVLayerSource]:
    result: list[StageVLayerSource] = []
    for layer in ("base", "meta"):
        for side in ("long", "short"):
            selector = _panel(layer=layer, side=side, start="2024-01-01", n=144, prefix="selector")
            oos = _panel(layer=layer, side=side, start="2024-01-08", n=48, prefix="oos")
            if collide_oos_with_oof:
                # Same side-qualified identity across strict OOF and later
                # OOS would let the map conflate two distinct observations.
                oos.loc[0, "candidate_id"] = selector.loc[0, "candidate_id"]
            # The later OOS starts strictly after the selector but has the
            # same selected raw contract and its own evaluation labels.
            if mutate_oos_target:
                oos.loc[:, f"{layer}_target"] = 1_000_000.0
            result.append(StageVLayerSource(
                layer=layer,
                side=side,
                selector=selector,
                oos=oos,
                raw_feature_names=_RAW,
                mda_group_audit=_audit() if positive_mda else _no_positive_audit(),
                target_column="r3_class" if layer == "base" else "exact_net_bps",
                selector_manifest_sha256="a" * 64,
                oos_surface_lineage={"declared": True, "surface": "synthetic-frozen-surface"},
            ))
    return result


def _config() -> StageVExperimentConfig:
    return StageVExperimentConfig(
        folds=3,
        min_train_rows=20,
        max_groups=4,
        min_selected_rows=4,
        admission_spec=Causal21dAdmissionSpec(
            min_reference_rows=12,
            min_side_reference_rows=4,
            side_shrinkage_rows=12.0,
            bins=4,
        ),
    )


def test_stage_v_writes_strict_oof_frozen_oos_and_causal_admission_artifacts(tmp_path) -> None:
    root = tmp_path / "stage_v"
    manifest = run_stage_v_drift_ood_ablation(sources=_sources(), output_dir=root, config=_config())

    assert manifest["schema"] == STAGE_V_EXPERIMENT_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["winner"]["promotion"] == "EXPERIMENT_WINNER_ONLY_NOT_POLICY_PROMOTION"
    assert manifest["winner"]["base_economics"] == "diagnostic_only_never_selectable"
    assert "direct_fold_local_FQ3" in manifest["architecture"]
    expected = {
        "joint_strict_oof_predictions.parquet",
        "joint_frozen_oos_predictions.parquet",
        "per_side_month_joint_meta_21d_metrics.parquet",
        "joint_strict_oof_with_causal_21d_admission.parquet",
        "joint_frozen_oos_with_causal_21d_admission.parquet",
        "causal_21d_admission_audit.parquet",
        "joint_feature_contracts.json",
        "winner.json",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in root.iterdir()})
    oof = pd.read_parquet(root / "joint_strict_oof_predictions.parquet")
    frozen = pd.read_parquet(root / "joint_frozen_oos_predictions.parquet")
    metrics = pd.read_parquet(root / "per_side_month_joint_meta_21d_metrics.parquet")
    mapped = pd.read_parquet(root / "joint_frozen_oos_with_causal_21d_admission.parquet")
    assert set(frozen.side_name) == {"long", "short"}
    assert {"control_meta_direct_score", "base_ood_meta_direct_score", "meta_ood_meta_direct_score", "both_ood_meta_direct_score"}.issubset(frozen.columns)
    assert {"strict_oof", "frozen_oos"}.issubset(metrics.population)
    assert "pooled_global_once_no_timestamp_or_side_rerank" in set(metrics.selection.dropna())
    assert "causal_21d_side_expected_net_bps" in mapped.columns
    assert mapped.candidate_key.str.contains("::").all()
    assert any(path.suffix == ".pkl" for path in (root / "models").rglob("*"))
    assert metrics.loc[metrics.layer.str.startswith("base_diagnostic"), "promotable"].eq(False).all()
    assert metrics.loc[metrics.layer.str.startswith("joint_meta"), "promotable"].eq(True).all()
    contracts = json.loads((root / "joint_feature_contracts.json").read_text())
    fq3 = contracts["long:both_ood:meta"]["fq3_features"]
    assert "base_raw_score" in fq3
    assert not any("mapped" in name or "expected_net" in name for name in fq3)


def test_frozen_oos_model_never_trains_on_oos_target_labels(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_stage_v_drift_ood_ablation(sources=_sources(), output_dir=first, config=_config())
    run_stage_v_drift_ood_ablation(sources=_sources(mutate_oos_target=True), output_dir=second, config=_config())
    a = pd.read_parquet(first / "joint_frozen_oos_predictions.parquet").sort_values("candidate_key")
    b = pd.read_parquet(second / "joint_frozen_oos_predictions.parquet").sort_values("candidate_key")
    for arm in ("control", "base_ood", "meta_ood", "both_ood"):
        assert np.allclose(a[f"{arm}_meta_direct_score"], b[f"{arm}_meta_direct_score"], rtol=0.0, atol=0.0)


def test_stage_v_rejects_undeclared_surface_lineage() -> None:
    source = _sources()[0]
    invalid = StageVLayerSource(
        **{**source.__dict__, "oos_surface_lineage": {"declared": False}},
    )
    with pytest.raises(StageVExperimentError, match="OOS-surface lineage"):
        invalid.validate()


def test_stage_v_rejects_the_old_independent_common_bps_base_route() -> None:
    source = _sources()[0]
    invalid = StageVLayerSource(**{**source.__dict__, "target_column": "exact_net_bps"})
    with pytest.raises(StageVExperimentError, match="native R3 classes"):
        invalid.validate()


def test_stage_v_rejects_mapped_or_expected_net_fq3_inputs() -> None:
    source = next(item for item in _sources() if item.layer == "meta")
    selector, oos = source.selector.copy(), source.oos.copy()
    selector["prequential_base_expected_net_bps"] = 0.0
    oos["prequential_base_expected_net_bps"] = 0.0
    invalid = StageVLayerSource(**{
        **source.__dict__,
        "selector": selector,
        "oos": oos,
        "raw_feature_names": (*source.raw_feature_names, "prequential_base_expected_net_bps"),
    })
    with pytest.raises(StageVExperimentError, match="mapped/expected-net"):
        invalid.validate()


def test_control_runs_when_frozen_mda_has_no_positive_group(tmp_path) -> None:
    root = tmp_path / "control_without_ood"
    run_stage_v_drift_ood_ablation(
        sources=_sources(positive_mda=False), output_dir=root, config=_config(),
    )
    prediction = pd.read_parquet(root / "joint_strict_oof_predictions.parquet")
    assert prediction.control_meta_direct_score.notna().any()
    assert prediction.both_ood_meta_direct_score.isna().all()
    contracts = json.loads((root / "joint_feature_contracts.json").read_text())
    assert contracts["long:control:base"]["state_fit"] == "not_requested"
    assert contracts["long:both_ood:base"]["availability"] == "no_positive_frozen_mda_group"


def test_combined_oof_oos_identity_collision_fails_before_causal_mapping(tmp_path) -> None:
    with pytest.raises(StageVExperimentError, match="canonical identity collision"):
        run_stage_v_drift_ood_ablation(
            sources=_sources(collide_oos_with_oof=True), output_dir=tmp_path / "collision", config=_config(),
        )


def test_control_best_retains_upstream_instead_of_promoting_stage_v() -> None:
    common = {
        "population": "strict_oof", "layer": "joint_meta:control",
        "admission_mode": "with_side_local_causal_21d_admission_after_reconstruction",
        "top_fraction": 0.10, "scope": "pooled_global", "candidate_rows": 100,
    }
    rows = [
        {**common, "arm": "control", "row_type": "pooled_global", "realised_net_bps_per_trade": 10.0, "selected_rows": 10},
        {**common, "layer": "joint_meta:both_ood", "arm": "both_ood", "row_type": "pooled_global", "realised_net_bps_per_trade": 8.0, "selected_rows": 10},
        {**common, "layer": "joint_meta:both_ood", "arm": "both_ood", "row_type": "selected_contribution", "scope": "month", "period_key": "2026-04", "realised_net_bps_per_trade": 8.0, "selected_rows": 10},
    ]
    result = _winner(pd.DataFrame(rows), _config())
    assert result["decision"] == "NO_STAGE_V_OOD_ADVANCE_RETAIN_UPSTREAM"
    assert result["winner_arm"] is None
