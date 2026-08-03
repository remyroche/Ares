from __future__ import annotations

import json

import pandas as pd

from scripts import summarize_root_cause_stage2_headline as headline


def _write_fixture(root) -> None:
    pd.DataFrame([
        {"feature_name": "alpha", "side": "long", "folds": 3, "transported_ic_mean": .11, "top_bottom_decile_spread_mean_bps": 4., "evaluated_rows": 100},
        {"feature_name": "beta", "side": "long", "folds": 3, "transported_ic_mean": .09, "top_bottom_decile_spread_mean_bps": 8., "evaluated_rows": 100},
        {"feature_name": "alpha", "side": "short", "folds": 3, "transported_ic_mean": .07, "top_bottom_decile_spread_mean_bps": 6., "evaluated_rows": 100},
    ]).to_parquet(root / "feature_information_results.parquet", index=False)
    pd.DataFrame([
        {"mechanism_group": "price_trend", "side": "long", "fold": "a", "status": "OK", "test_rows": 10, "spearman_ic": .1, "top_bottom_decile_spread_bps": 3., "oof_mae_bps": 12.},
        {"mechanism_group": "price_trend", "side": "long", "fold": "b", "status": "NOT_RUN", "test_rows": 0, "spearman_ic": None, "top_bottom_decile_spread_bps": None, "oof_mae_bps": None},
    ]).to_parquet(root / "feature_information_mechanism_oof.parquet", index=False)
    pd.DataFrame([
        {"head": "canonical_gross_base", "probe_family": "same", "side": "long", "fold": "a", "status": "OK", "test_rows": 10, "residual_probe_oof_ic": .2, "residual_probe_oof_mae_bps": 9.},
        {"head": "canonical_gross_base", "probe_family": "same", "side": "long", "fold": "b", "status": "NOT_RUN", "test_rows": 0, "residual_probe_oof_ic": None, "residual_probe_oof_mae_bps": None},
    ]).to_parquet(root / "feature_information_residual_probes.parquet", index=False)
    pd.DataFrame([
        {"side": "long", "psi": .1, "jensen_shannon": .01, "wasserstein": 2., "missingness_delta": -.02},
        {"side": "long", "psi": .3, "jensen_shannon": .03, "wasserstein": 4., "missingness_delta": .04},
    ]).to_parquet(root / "feature_information_drift.parquet", index=False)
    pd.DataFrame([
        {"scope_type": "side_month", "side": "long", "adversarial_auc": .61},
        {"scope_type": "side_month", "side": "long", "adversarial_auc": .73},
    ]).to_parquet(root / "feature_information_cohort_drift.parquet", index=False)
    pd.DataFrame([
        {"research_causal_probe_eligible": True, "production_live_reuse_eligible": False, "live_reproducibility_status": "NOT_VERIFIED", "staleness_status": "NOT_VERIFIED"},
        {"research_causal_probe_eligible": True, "production_live_reuse_eligible": True, "live_reproducibility_status": "VERIFIED", "staleness_status": "VERIFIED"},
    ]).to_parquet(root / "feature_information_inventory.parquet", index=False)
    pd.DataFrame([{"status": "OK"}, {"status": "NOT_RUN"}]).to_parquet(root / "feature_information_fold_local_gross_mapping.parquet", index=False)
    pd.DataFrame([{"status": "OK"}, {"status": "NOT_RUN"}]).to_parquet(root / "feature_information_current_netmap_diagnostics.parquet", index=False)
    (root / "run_manifest.json").write_text(json.dumps({"inputs_sha256": {"ledger": "abc"}, "outputs_sha256": {"x": "def"}}))


def test_headline_summary_is_deterministic_and_preserves_availability_distinction(tmp_path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _write_fixture(source)
    first, payload = headline.build_headline(source)
    second, _ = headline.build_headline(source)
    assert first.equals(second)
    best_long_ic = first.loc[(first.section.eq("best_transported_feature")) & (first.metric.eq("transported_ic")) & (first.side.eq("long"))].iloc[0]
    best_long_spread = first.loc[(first.section.eq("best_transported_feature")) & (first.metric.eq("top_bottom_decile_spread_bps")) & (first.side.eq("long"))].iloc[0]
    assert best_long_ic.feature_name == "alpha"
    assert best_long_spread.feature_name == "beta"
    availability = first.loc[first.section.eq("availability")].set_index("metric")
    assert availability.loc["research_causal_cutoff_verified", "value"] == 2.0
    assert availability.loc["production_live_reuse_verified", "value"] == 1.0
    support = first.loc[(first.section.eq("support_status")) & (first.source_artifact.eq("residual_probes"))].set_index("status")
    assert support.loc["NOT_RUN", "value"] == 1.0
    assert "research_causal_cutoff_verified" in payload["availability_interpretation"]


def test_headline_writer_emits_json_and_parquet(tmp_path) -> None:
    source = tmp_path / "source"; source.mkdir()
    output = tmp_path / "output"; output.mkdir()
    _write_fixture(source)
    result = headline.run(input_dir=source, output_dir=output)
    assert result["rows"] > 0
    assert (output / headline.DEFAULT_PARQUET).is_file()
    payload = json.loads((output / headline.DEFAULT_JSON).read_text())
    assert payload["schema"] == headline.SCHEMA
    assert payload["status"].startswith("READ_ONLY")
