from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_asset_representation_meta_ablation_v2 import (  # noqa: E402
    AE_REPRESENTATION_COLUMNS,
    CONDITIONAL_REPRESENTATION_COLUMNS,
    CROSS_LGBM_REPRESENTATION_COLUMNS,
    REPRESENTATION_COLUMNS,
    main,
)


def _build_handoff_ledger_and_reps() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(99)
    handoff_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    rep_rows: list[dict[str, object]] = []
    for month_idx, month in enumerate(("2026-04", "2026-05", "2026-06")):
        base_ts = pd.Timestamp(f"{month}-01 00:00:00")
        for idx in range(180):
            side = "long" if idx % 4 == 0 else "short"
            symbol = f"SYM{idx % 9}"
            ts = base_ts + pd.Timedelta(hours=idx)
            clean_signal = np.sin(idx / 9.0) + 0.2 * (idx % 5) - 0.15 * month_idx
            dirty_signal = -clean_signal + rng.normal(0.0, 0.2)
            clean = clean_signal > 0.10
            bad_mae = dirty_signal > 0.65
            timeout = idx % 31 == 0 and not clean
            exec_margin = 0.010 if clean and not bad_mae else -0.006
            source = "run_entry" if idx % 3 else "late_run_continuation"
            handoff_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": float(clean_signal + rng.normal(0.0, 0.3)),
                    "selected_top10": True,
                    "source_semantic_family": source,
                    "source_semantic_family_base": source,
                    "long_source_regime_split": source if side == "long" else "not_long",
                    "aegmm_cluster": f"aegmm_cluster__{idx % 3}",
                    "side_aegmm_cluster": f"{side}__{idx % 3}",
                    "aegmm_expected_distance_bin": "near" if clean else "far",
                    "regime_clean_exec_score": 0.8 if clean else 0.2,
                    "regime_bad_mae_score": 0.2 if clean else 0.8,
                    "gmm_entropy": float(abs(dirty_signal)),
                    "gmm_posterior_max": float(0.60 + 0.01 * (idx % 7)),
                    "latent_speed": float(clean_signal),
                    "AE_reconstruction_error": float(abs(dirty_signal)),
                    "meta_context_weight_hint": 0.8 if clean else 0.2,
                    "meta_threshold_adjustment_hint": 0.1 if clean else 0.7,
                }
            )
            ledger_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": float(clean_signal),
                    "selected_top10": True,
                    "exec_margin": float(exec_margin),
                    "ev_after_1pct": float(exec_margin - 0.01),
                    "ret_net": float(exec_margin - 0.01),
                    "first_touch_gross": float(exec_margin + 0.01),
                    "first_touch_bad_mae_1r": float(bad_mae),
                    "full_path_bad_mae_1r": float(bad_mae),
                    "timeout": float(timeout),
                    "mfe_before_mae_1r": float(clean and not bad_mae),
                    "mae_before_mfe_1r": float(bad_mae),
                    "clean_exec": float(exec_margin > 0.0 and not bad_mae and not timeout),
                    "dirty_positive": float(exec_margin > 0.0 and (bad_mae or timeout)),
                    "underwater_bars_before_mfe_1r": float(2 if clean else 12),
                }
            )
            if month != "2026-04":
                rep = {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "cross_lgbm_exec_margin_score": float(clean_signal),
                    "cross_lgbm_bad_mae_score": float(1.0 / (1.0 + np.exp(-dirty_signal))),
                    "cross_lgbm_timeout_score": float(0.9 if timeout else 0.1),
                    "cross_lgbm_dirty_positive_score": float(0.8 if exec_margin > 0.0 and (bad_mae or timeout) else 0.2),
                    "cross_lgbm_clean_risk_composite": float(clean_signal - dirty_signal),
                    "market_z_0": float(clean_signal),
                    "market_z_1": float(dirty_signal),
                    "market_z_2": float(clean_signal * dirty_signal),
                    "market_z_3": float(month_idx),
                    "market_ae_recon_error": float(abs(dirty_signal)),
                    "market_ae_recon_error_pct": float(min(1.0, abs(dirty_signal))),
                    "market_ae_mahalanobis_diag": float(dirty_signal * dirty_signal),
                }
                for family in (
                    "tail",
                    "breadth",
                    "dispersion",
                    "btc_eth",
                    "corr_spectral",
                    "xasset",
                    "gmm_ae",
                    "regime_context",
                    "other",
                ):
                    rep[f"family_recon_error_{family}"] = float(abs(dirty_signal) + 0.01 * (idx % 3))
                rep_rows.append(rep)
    return pd.DataFrame(handoff_rows), pd.DataFrame(ledger_rows), pd.DataFrame(rep_rows)


def test_cross_asset_representation_meta_ablation_v2_strict_oof_contract(tmp_path: Path) -> None:
    handoff, ledger, reps = _build_handoff_ledger_and_reps()
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    handoff.to_parquet(handoff_dir / "train_meta_regime_handoff.parquet", index=False)
    ledger.to_parquet(handoff_dir / "s52_trailing_regime_scored_ledger.parquet", index=False)
    rep_path = tmp_path / "representations.parquet"
    reps.to_parquet(rep_path, index=False)

    rc = main(
        [
            "--handoff-dir",
            str(handoff_dir),
            "--representation-predictions",
            str(rep_path),
            "--out-dir",
            str(out_dir),
            "--frontier",
            "10",
            "--train-scope",
            "selected",
            "--min-train-rows",
            "20",
            "--min-valid-rows",
            "20",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["scored_months"] == ["2026-06"]
    assert manifest["rows_with_oof_representation"] == 360
    baseline_features = set(manifest["feature_columns_by_variant"]["m0_baseline_meta"])
    augmented_features = set(manifest["feature_columns_by_variant"]["m1_cross_lgbm_meta"])
    risk_only_features = set(manifest["feature_columns_by_variant"]["m1b_cross_lgbm_risk_only_meta"])
    badmae_only_features = set(manifest["feature_columns_by_variant"]["m1c_cross_lgbm_badmae_only_meta"])
    ae_features = set(manifest["feature_columns_by_variant"]["m2_market_ae_ood_meta"])
    combined_features = set(manifest["feature_columns_by_variant"]["m3_cross_lgbm_plus_ae_meta"])
    conditional_cross_features = set(manifest["feature_columns_by_variant"]["m4_conditional_cross_meta"])
    conditional_ae_features = set(manifest["feature_columns_by_variant"]["m5_conditional_ae_meta"])
    conditional_combined_features = set(manifest["feature_columns_by_variant"]["m6_conditional_cross_plus_ae_meta"])
    raw_plus_conditional_features = set(manifest["feature_columns_by_variant"]["m7_cross_plus_conditional_cross_meta"])
    assert baseline_features.isdisjoint(REPRESENTATION_COLUMNS)
    assert baseline_features.isdisjoint(CONDITIONAL_REPRESENTATION_COLUMNS)
    assert set(CROSS_LGBM_REPRESENTATION_COLUMNS).issubset(augmented_features)
    assert augmented_features.isdisjoint(AE_REPRESENTATION_COLUMNS)
    assert "cross_lgbm_exec_margin_score" not in risk_only_features
    assert "cross_lgbm_bad_mae_score" in risk_only_features
    assert "cross_lgbm_bad_mae_score" in badmae_only_features
    assert "cross_lgbm_timeout_score" not in badmae_only_features
    assert set(AE_REPRESENTATION_COLUMNS).issubset(ae_features)
    assert ae_features.isdisjoint(CROSS_LGBM_REPRESENTATION_COLUMNS)
    assert set(REPRESENTATION_COLUMNS).issubset(combined_features)
    assert any(col.startswith("cond_cross_") for col in conditional_cross_features)
    assert not any(col.startswith("cond_market_ae_") for col in conditional_cross_features)
    assert any(col.startswith("cond_market_ae_") for col in conditional_ae_features)
    assert not any(col.startswith("cond_cross_") for col in conditional_ae_features)
    assert any(col.startswith("cond_cross_") for col in conditional_combined_features)
    assert any(col.startswith("cond_market_ae_") for col in conditional_combined_features)
    assert set(CROSS_LGBM_REPRESENTATION_COLUMNS).issubset(raw_plus_conditional_features)
    assert any(col.startswith("cond_cross_") for col in raw_plus_conditional_features)
    assert augmented_features.isdisjoint(CONDITIONAL_REPRESENTATION_COLUMNS)
    assert manifest["conditional_acceptance_by_scored_month"]
    assert manifest["conditional_acceptance_rows"] > 0
    assert "promotion_recommendation" in manifest
    assert "exec_margin" not in augmented_features
    assert "long_bad_path_label" not in augmented_features
    assert "has_cross_lgbm_representation" not in augmented_features
    assert "gmm_cluster_id" not in augmented_features
    assert "aegmm_cluster" not in augmented_features
    summary = pd.read_csv(out_dir / "cross_asset_representation_meta_ablation_v2_summary.csv")
    assert not summary.empty
    assert summary["selector"].astype(str).str.startswith("m0_baseline_meta").any()
    assert summary["selector"].astype(str).str.startswith("m1_cross_lgbm_meta").any()
    assert summary["selector"].astype(str).str.startswith("m7_cross_plus_conditional_cross_meta").any()
    conditional = pd.read_csv(out_dir / "cross_asset_representation_meta_ablation_v2_conditional_acceptance.csv")
    assert not conditional.empty
    assert {"control_adjusted_exec_margin", "accepted", "delta_bad_mae"}.issubset(conditional.columns)
    promotion = json.loads((out_dir / "cross_asset_representation_meta_ablation_v2_promotion.json").read_text())
    assert "promote_to_deeper_meta_eval" in promotion
    assert "shadow_only" in promotion
    preds = pd.read_parquet(out_dir / "cross_asset_representation_meta_ablation_v2_predictions.parquet")
    assert set(preds["month"].astype(str).unique()) == {"2026-06"}
