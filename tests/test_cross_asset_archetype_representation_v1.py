from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cross_asset_archetype_representation_v1 import AE_OUTPUT_COLUMNS, main


def _synthetic_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    months = ["2026-04", "2026-05", "2026-06"]
    symbols = [f"SYM{i}/USD:USD" for i in range(6)]
    for month_idx, month in enumerate(months):
        base_ts = pd.Timestamp(f"{month}-01 00:00:00")
        for hour in range(24):
            for sym_idx, symbol in enumerate(symbols):
                side = "long" if (hour + sym_idx) % 2 == 0 else "short"
                source = "compression_release" if (hour + sym_idx) % 3 else "retest_reversal"
                signal = np.sin(hour / 4.0) + 0.4 * sym_idx - 0.25 * month_idx
                cross = signal + rng.normal(0.0, 0.15)
                risk = -signal + rng.normal(0.0, 0.20)
                exec_margin = 0.006 * cross - 0.003 * (side == "long") + rng.normal(0.0, 0.002)
                bad_mae = float((risk > 0.8) or (side == "long" and cross < 0.1))
                timeout = float((hour % 7 == 0) and cross < 1.0)
                clean_exec = float(exec_margin > 0.0 and bad_mae < 0.5 and timeout < 0.5)
                dirty_positive = float(exec_margin > 0.0 and (bad_mae > 0.5 or timeout > 0.5))
                rows.append(
                    {
                        "__ts__": base_ts + pd.Timedelta(hours=hour),
                        "__symbol__": symbol,
                        "side_name": side,
                        "month": month,
                        "score": float(cross + rng.normal(0.0, 0.3)),
                        "selected_top10": True,
                        "source_family": source,
                        "source_semantic_family": source,
                        "source_semantic_family_base": source,
                        "long_source_regime_split": source if side == "long" else "not_long",
                        "aegmm_cluster": f"aegmm_cluster__{sym_idx % 3}",
                        "side_aegmm_cluster": f"{side}__{sym_idx % 3}",
                        "aegmm_entropy_bin": "low" if sym_idx % 2 else "high",
                        "aegmm_distance_bin": "near" if sym_idx % 2 else "far",
                        "reconstruction_bin": "ok" if cross > 0 else "stressed",
                        "gmm_cluster_id": float(sym_idx % 3),
                        "gmm_posterior_max": float(0.55 + 0.05 * (sym_idx % 3)),
                        "gmm_entropy": float(abs(risk) / 4.0),
                        "mahalanobis_distance": float(abs(risk)),
                        "AE_reconstruction_error": float(abs(rng.normal(0.2, 0.05)) + 0.1 * bad_mae),
                        "cluster_speed": float(rng.normal(0.0, 0.2)),
                        "cluster_acceleration": float(rng.normal(0.0, 0.1)),
                        "q_tail_market_pressure": float(cross),
                        "mkt_breadth_pressure": float(cross * 0.5),
                        "state_spectral_eig_gap": float(1.0 / (1.0 + abs(risk))),
                        "exec_margin": float(exec_margin),
                        "ev_after_1pct": float(exec_margin - 0.01),
                        "ret_net": float(exec_margin - 0.01),
                        "clean_exec": clean_exec,
                        "dirty_positive": dirty_positive,
                        "full_path_bad_mae_1r": bad_mae,
                        "first_touch_bad_mae_1r": bad_mae,
                        "timeout": timeout,
                        "mfe_before_mae_1r": float(clean_exec),
                        "mae_before_mfe_1r": float(bad_mae),
                        "max_adverse_before_mfe_1r": float(abs(risk)),
                        "underwater_bars_before_mfe_1r": float(max(0.0, risk * 4.0)),
                    }
                )
    frame = pd.DataFrame(rows)
    ledger_cols = [
        "__ts__",
        "__symbol__",
        "side_name",
        "month",
        "score",
        "selected_top10",
        "exec_margin",
        "ev_after_1pct",
        "ret_net",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "first_touch_bad_mae_1r",
        "timeout",
        "mfe_before_mae_1r",
        "mae_before_mfe_1r",
        "max_adverse_before_mfe_1r",
        "underwater_bars_before_mfe_1r",
    ]
    handoff = frame.drop(columns=["exec_margin", "ev_after_1pct", "ret_net", "clean_exec", "dirty_positive", "full_path_bad_mae_1r", "first_touch_bad_mae_1r", "timeout", "mfe_before_mae_1r", "mae_before_mfe_1r", "max_adverse_before_mfe_1r", "underwater_bars_before_mfe_1r"])
    ledger = frame[ledger_cols].copy()
    return handoff, ledger


def test_cross_asset_representation_v1_month_forward_and_feature_contract(tmp_path: Path) -> None:
    handoff, ledger = _synthetic_rows()
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    handoff.to_parquet(handoff_dir / "train_meta_regime_handoff.parquet", index=False)
    ledger.to_parquet(handoff_dir / "s52_trailing_regime_scored_ledger.parquet", index=False)

    rc = main(
        [
            "--handoff-dir",
            str(handoff_dir),
            "--out-dir",
            str(out_dir),
            "--frontier",
            "10",
            "--train-scope",
            "selected",
            "--min-fold-train-rows",
            "20",
            "--min-fold-valid-rows",
            "20",
            "--min-train-cell-rows",
            "10",
            "--min-valid-cell-rows",
            "5",
            "--min-train-clean-rows",
            "1",
            "--min-valid-clean-rows",
            "1",
            "--max-single-asset-share",
            "0.90",
            "--max-single-week-share",
            "1.00",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["model_a_status"] == "implemented"
    assert manifest["model_b_status"] == "implemented_compact_linear_denoising_ae"
    assert manifest["scored_months"] == ["2026-05", "2026-06"]
    assert set(AE_OUTPUT_COLUMNS).issubset(set(manifest["ae_output_columns"]))
    assert "exec_margin" not in manifest["feature_columns"]
    assert "full_path_bad_mae_1r" not in manifest["feature_columns"]
    assert "gmm_cluster_id" not in manifest["feature_columns"]
    predictions = pd.read_parquet(out_dir / "cross_asset_representation_v1_predictions.parquet")
    assert {"cross_lgbm_exec_margin_score", "cross_lgbm_bad_mae_score", "cross_lgbm_clean_risk_composite"}.issubset(predictions.columns)
    assert {"market_z_0", "market_ae_recon_error", "market_ae_recon_error_pct", "family_recon_error_gmm_ae"}.issubset(predictions.columns)
    assert predictions["market_ae_recon_error"].notna().any()
    assert set(predictions["month"].astype(str).unique()) == {"2026-05", "2026-06"}
    cells = pd.read_csv(out_dir / "cross_asset_representation_v1_cell_diagnostics.csv")
    assert {"supported_cell", "control_adjusted_exec_margin", "delta_bad_mae_vs_base"}.issubset(cells.columns)
