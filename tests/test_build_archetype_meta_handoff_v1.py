from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_archetype_meta_handoff_v1 import build_handoff


def _synthetic_ledger() -> pd.DataFrame:
    rows = []
    months = ["2026-01", "2026-02", "2026-03"]
    for mi, month in enumerate(months):
        for i in range(80):
            side = "long" if i % 2 == 0 else "short"
            clean_group = i % 4 in {0, 1}
            score = 0.8 if clean_group else 0.55
            if mi == 2 and i % 10 == 0:
                score = 0.95
                clean_group = False
            bad = 0 if clean_group else 1
            timeout = 1 if (not clean_group and i % 7 == 0) else 0
            exec_margin = 0.02 if clean_group else -0.01
            dirty = 1 if (not clean_group and i % 3 == 0) else 0
            rows.append(
                {
                    "variant": "synthetic",
                    "month": month,
                    "__ts__": pd.Timestamp(f"{month}-01") + pd.Timedelta(hours=i),
                    "__symbol__": f"SYM{i % 12}",
                    "side_name": side,
                    "score": score + 0.001 * (i % 5),
                    "side": 1.0 if side == "long" else -1.0,
                    "__regime_vol_12h__": float(i % 4) + (0.5 if clean_group else 4.0),
                    "__regime_vol_48h__": float(i % 5) + (0.3 if clean_group else 3.0),
                    "__meta_raw__volatility_zscore": float(i % 6) + (0.1 if clean_group else 2.5),
                    "G_VOL": float(i % 3),
                    "selected_top10": 1 if score > 0.75 else 0,
                    "exec_margin": exec_margin,
                    "ret_net": exec_margin + 0.004,
                    "u_policy_net": exec_margin + 0.007,
                    "clean_exec": 1 if clean_group else 0,
                    "dirty_positive": dirty,
                    "full_path_bad_mae_1r": bad,
                    "first_touch_bad_mae_1r": bad,
                    "timeout": timeout,
                    "mae_norm": 0.2 if clean_group else 1.2,
                    "mfe_norm": 1.2 if clean_group else 0.5,
                    "underwater_bars_before_mfe_1r": 2 if clean_group else 15,
                }
            )
    return pd.DataFrame(rows)


def test_archetype_meta_handoff_v1_writes_oof_artifacts(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    _synthetic_ledger().to_parquet(ledger_path, index=False)

    manifest = build_handoff(
        ledger_path=ledger_path,
        out_dir=tmp_path / "out",
        n_components=4,
        seeds=(3, 5),
        min_train_months=1,
        n_latent=3,
        shrinkage_k=20.0,
        top_frac=0.10,
    )

    assert manifest["acceptance_tests"]["leakage_test"]["status"] == "pass"
    assert manifest["oof_validation_rows"] == 160
    rows = pd.read_parquet(manifest["outputs"]["row_features"])
    profiles = pd.read_parquet(manifest["outputs"]["profile_table"])
    reliability = pd.read_parquet(manifest["outputs"]["reliability_table"])
    ablation = pd.read_csv(manifest["outputs"]["handoff_ablation"])

    assert not rows.empty
    assert not profiles.empty
    assert not reliability.empty
    assert set(rows["month"].astype(str)) == {"2026-02", "2026-03"}
    assert {"gmm_posterior_0", "gmm_entropy_oof", "ae_reconstruction_error_oof"}.issubset(rows.columns)
    assert {"prior_base_clean_rate", "prior_base_bad_MAE_rate", "shrunk_base_clean_rate"}.issubset(rows.columns)
    assert {"base_overconfident_bad_MAE", "base_utility_residual"}.issubset(rows.columns)
    assert set(ablation["score_model"]) == {
        "score_M0_no_archetype",
        "score_M1_hard_archetype_id",
        "score_M2_gmm_posterior_entropy",
        "score_M3_archetype_outcome_priors",
        "score_M4_base_reliability_priors",
    }
    leakage = pd.read_csv(manifest["outputs"]["leakage_report"])
    assert leakage["scaler_fit_scope"].eq("outer_train_only").all()
    assert leakage["validation_assignment_scope"].eq("frozen_train_artifacts").all()
