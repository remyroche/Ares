from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_path_quality_archetype_ablation import SCORE_COLUMNS, run_ablation


def _synthetic_ledger() -> pd.DataFrame:
    rows = []
    months = ["2026-01", "2026-02", "2026-03"]
    for mi, month in enumerate(months):
        for i in range(120):
            side = "long" if i % 3 == 0 else "short"
            clean = i % 5 in {0, 1}
            bad = not clean and i % 4 != 0
            timeout = not clean and i % 11 == 0
            score = 0.78 if clean else 0.55
            rows.append(
                {
                    "month": month,
                    "__ts__": pd.Timestamp(f"{month}-01") + pd.Timedelta(minutes=15 * i),
                    "__symbol__": f"S{i % 16}",
                    "side_name": side,
                    "score": score + 0.001 * (i % 7) + 0.01 * mi,
                    "side": 1.0 if side == "long" else -1.0,
                    "__regime_vol_12h__": float(i % 4) + (0.1 if clean else 2.0),
                    "__regime_vol_48h__": float(i % 5) + (0.2 if clean else 2.5),
                    "__regime_volume_12h__": float(i % 6),
                    "__regime_trend_12h__": float(i % 3) - (0.5 if clean else 0.0),
                    "__meta_raw__volatility_zscore": float(i % 8) + (0.1 if clean else 3.0),
                    "G_VOL": float(i % 4),
                    "aegmm_cluster": f"c{i % 3}",
                    "side_aegmm_cluster": f"{side}__c{i % 3}",
                    "reconstruction_bin": "low" if clean else "high",
                    "cluster_speed_bin": f"speed_{i % 4}",
                    "source_semantic_family": "clean_family" if clean else "dirty_family",
                    "source_volatility_state": "low_volatility" if clean else "high_volatility",
                    "source_pressure_state": f"p{i % 2}",
                    "source_trend_state": f"t{i % 3}",
                    "source_score_intensity_tag": f"{side}__tag",
                    "selected_top10": int(score > 0.75),
                    "exec_margin": 0.02 if clean else -0.008,
                    "clean_exec": int(clean),
                    "dirty_positive": int((not clean) and i % 6 == 0),
                    "full_path_bad_mae_1r": int(bad),
                    "timeout": int(timeout),
                }
            )
    return pd.DataFrame(rows)


def test_path_quality_archetype_ablation_writes_oof_outputs(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.parquet"
    _synthetic_ledger().to_parquet(ledger_path, index=False)

    manifest = run_ablation(
        ledger_path=ledger_path,
        out_dir=tmp_path / "out",
        min_train_months=1,
        seed=7,
        top_frac=0.10,
    )

    assert manifest["acceptance_tests"]["leakage_test"]["status"] == "pass"
    assert manifest["oof_validation_rows"] == 240
    rows = pd.read_parquet(manifest["outputs"]["row_features"])
    summary = pd.read_csv(manifest["outputs"]["summary"])
    leakage = pd.read_csv(manifest["outputs"]["leakage"])

    assert set(rows["month"].astype(str)) == {"2026-02", "2026-03"}
    assert set(SCORE_COLUMNS).issubset(rows.columns)
    assert {"global_path_clean", "side_path_bad", "predicted_path_archetype"}.issubset(rows.columns)
    assert set(summary["score_model"]) == set(SCORE_COLUMNS)
    assert leakage["model_fit_scope"].eq("outer_train_only").all()
    assert leakage["validation_assignment_scope"].eq("frozen_train_models_and_priors").all()
