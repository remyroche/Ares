from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.feature_portability import FeaturePortabilityError, PortabilityPolicy
from extreme_price_movements.feature_portability_audit import (
    ChronologicalAuditPolicy,
    population_stability_index,
    run_chronological_feature_portability_audit,
    wasserstein_distance_1d,
    write_feature_portability_artifacts,
)


def _three_era_panel(*, late_target_sign: float = 1.0, late_distribution: str = "same") -> pd.DataFrame:
    values = np.tile(np.arange(10, dtype=float), 10)
    if late_distribution == "era_shortcut":
        # Every reference bin remains represented, but 91% of the late era
        # occupies the top bin.  This separates eras without extrapolating.
        late_values = np.concatenate([np.arange(9, dtype=float), np.full(91, 9.0)])
    else:
        late_values = values.copy()
    blocks = []
    for offset, era, feature, sign in (
        (0, "early", values, 1.0),
        (200, "middle", values, 1.0),
        (400, "late", late_values, late_target_sign),
    ):
        blocks.append(
            pd.DataFrame(
                {
                    "ts": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC") + pd.Timedelta(hours=offset),
                    "era": era,
                    "side": "long",
                    "ret_1h": feature,
                    "target": sign * feature,
                    "economic_residual_bps": sign * (feature - 4.5),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _policy() -> ChronologicalAuditPolicy:
    return ChronologicalAuditPolicy(
        portability=PortabilityPolicy(
            min_coverage=0.99,
            min_finite_support=50,
            min_unique_values=2,
            max_extrapolation_rate=0.01,
            min_bin_support=1,
            min_effect_support=50,
        ),
        min_reference_rows=50,
        distribution_bins=10,
        max_era_shortcut_auc=0.65,
        min_semantic_stability=0.50,
    )


def test_chronological_audit_uses_only_prior_rows_and_emits_stage_a_metrics() -> None:
    result = run_chronological_feature_portability_audit(
        _three_era_panel(),
        feature_names=["ret_1h"],
        timestamp_column="ts",
        era_column="era",
        strata_columns=["side"],
        target_column="target",
        economic_residual_column="economic_residual_bps",
        policy=_policy(),
    )
    era = result.era_audit.set_index("era")
    assert bool(era.loc["early", "reference_ready"]) is False
    assert int(era.loc["middle", "reference_rows"]) == 100
    assert int(era.loc["late", "reference_rows"]) == 200
    assert era.loc["middle", "coverage"] == pytest.approx(1.0)
    assert era.loc["middle", "extrapolation_rate"] == pytest.approx(0.0)
    assert era.loc["middle", "current_bin_min_support"] >= 1
    assert era.loc["middle", "psi"] == pytest.approx(0.0)
    assert era.loc["middle", "wasserstein"] == pytest.approx(0.0)
    assert era.loc["middle", "robust_median_shift"] == pytest.approx(0.0)
    assert era.loc["middle", "semantic_stability_proxy"] == pytest.approx(1.0)
    assert era.loc["middle", "economic_residual_spearman"] > 0.99
    assert result.dispositions.loc[0, "disposition"] == "INVARIANT_RELATIVE"
    assert result.manifest["latent_regime_outputs_allowed"] is False


def test_era_shortcut_and_semantic_instability_have_distinct_dispositions() -> None:
    shortcut = run_chronological_feature_portability_audit(
        _three_era_panel(late_distribution="era_shortcut"),
        feature_names=["ret_1h"], timestamp_column="ts", era_column="era",
        target_column="target", policy=_policy(),
    )
    assert shortcut.era_audit.loc[shortcut.era_audit["era"].eq("late"), "era_shortcut_auc"].iloc[0] > 0.80
    assert shortcut.dispositions.loc[0, "disposition"] == "ERA_SHORTCUT"

    unstable = run_chronological_feature_portability_audit(
        _three_era_panel(late_target_sign=-1.0),
        feature_names=["ret_1h"], timestamp_column="ts", era_column="era",
        target_column="target", economic_residual_column="economic_residual_bps", policy=_policy(),
    )
    late = unstable.era_audit.loc[unstable.era_audit["era"].eq("late")].iloc[0]
    assert late["semantic_stability_proxy"] < 0.5
    assert late["economic_residual_spearman"] < -0.99
    assert unstable.dispositions.loc[0, "disposition"] == "UNSTABLE"


def test_99_percent_coverage_extrapolation_and_bin_support_gates_fail_closed() -> None:
    coverage = _three_era_panel()
    coverage.loc[coverage.index[-2:], "ret_1h"] = np.nan
    coverage_result = run_chronological_feature_portability_audit(
        coverage, feature_names=["ret_1h"], timestamp_column="ts", era_column="era", policy=_policy()
    )
    assert coverage_result.dispositions.loc[0, "disposition"] == "UNSTABLE"
    assert "coverage" in coverage_result.dispositions.loc[0, "disposition_reason"]

    extrapolated = _three_era_panel()
    extrapolated.loc[extrapolated.index[-100:], "ret_1h"] = np.arange(100, 200, dtype=float)
    extrapolated_result = run_chronological_feature_portability_audit(
        extrapolated, feature_names=["ret_1h"], timestamp_column="ts", era_column="era", policy=_policy()
    )
    assert extrapolated_result.dispositions.loc[0, "disposition"] == "UNSTABLE"
    assert "extrapolation" in extrapolated_result.dispositions.loc[0, "disposition_reason"]

    missing_bins = _three_era_panel()
    missing_bins.loc[missing_bins.index[-100:], "ret_1h"] = 9.0
    bins_result = run_chronological_feature_portability_audit(
        missing_bins, feature_names=["ret_1h"], timestamp_column="ts", era_column="era", policy=_policy()
    )
    assert bins_result.dispositions.loc[0, "disposition"] == "UNSTABLE"
    assert "bin-support" in bins_result.dispositions.loc[0, "disposition_reason"]


def test_latent_regime_inputs_are_rejected_and_artifacts_include_manifest(tmp_path) -> None:
    frame = _three_era_panel()
    frame["gmm_posterior_0"] = 0.5
    with pytest.raises(FeaturePortabilityError, match="forbids latent"):
        run_chronological_feature_portability_audit(
            frame, feature_names=["gmm_posterior_0"], timestamp_column="ts", era_column="era", policy=_policy()
        )
    result = run_chronological_feature_portability_audit(
        frame, feature_names=["ret_1h"], timestamp_column="ts", era_column="era", policy=_policy()
    )
    paths = write_feature_portability_artifacts(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema"] == "stage_a_feature_portability_audit_v1"
    assert manifest["features"][0]["disposition"] == "INVARIANT_RELATIVE"


def test_distribution_metrics_are_exact_on_small_arrays() -> None:
    assert wasserstein_distance_1d(np.array([0.0, 1.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)
    assert population_stability_index(np.arange(10, dtype=float), np.arange(10, dtype=float)) == pytest.approx(0.0)
