#!/usr/bin/env python3
"""Fit the declared frozen F0 R3 base model and score later candidates.

This is intentionally separate from the GAM replay.  It makes the frozen
F0 model artifact explicit, joins only the exact later TP6/SL4 labels, and
fails if any declared feature is absent from the regenerated context panel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_feature_leaf_reasoning_portability import _r3_weights  # noqa: E402
from extreme_price_movements.tp6_portability_data import TP6_SL4_COST_BPS  # noqa: E402


CONTRACT = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_representation_contracts.json"
MODEL_CONTRACT = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_frozen_r3_model_contract.json"
HISTORICAL = ROOT / "data_perp/artifacts/feature_portability_f4_panel_20260804_v1/f4_candidate_panel.parquet"
LATER_CONTEXT = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/rebuilt_f0_context_full_repair_v2/later_f0_context.parquet"
LABEL_DIR = ROOT / "data_perp/artifacts/tp6_sl4_gam_untouched_later_20260815_v1/assembled_exact_labels/parts/month=2026-07"
OUT = ROOT / "data_perp/artifacts/tp6_sl4_frozen_f0_later_base_20260808_v1"


def _matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"F0 matrix missing fields: {missing}")
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").astype("float32")


def _fit(frame: pd.DataFrame, fields: list[str], seed: int) -> lgb.LGBMClassifier:
    if len(frame) < 10_000:
        raise ValueError(f"historical side support too small: {len(frame)}")
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, random_state=seed,
        n_estimators=140, learning_rate=0.05, num_leaves=31,
        min_child_samples=350, subsample=0.80, colsample_bytree=0.80,
        reg_lambda=8.0, n_jobs=1, verbosity=-1,
    )
    model.fit(_matrix(frame, fields), frame["r3_class"].to_numpy(np.int8), sample_weight=_r3_weights(frame))
    if not np.array_equal(np.asarray(model.classes_, dtype=np.int8), np.array([0, 1, 2], dtype=np.int8)):
        raise ValueError("frozen class order is not adverse=0, weak=1, clear=2")
    return model


def run() -> Path:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text())["F0_current_frozen"]
    hist = pd.read_parquet(HISTORICAL)
    later = pd.read_parquet(LATER_CONTEXT)
    labels = pd.concat([pd.read_parquet(LABEL_DIR / "side=long.parquet"), pd.read_parquet(LABEL_DIR / "side=short.parquet")], ignore_index=True)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True)
    later["__ts__"] = pd.to_datetime(later["__ts__"], utc=True)
    later = later.merge(labels[["candidate_id", "__ts__", "side_name", "__label_available_at__", "label_valid", "target_invalid", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "robust_clear_event_b25"]], on=["candidate_id", "__ts__", "side_name"], how="left", validate="one_to_one")
    if later.candidate_id.isna().any() or later.candidate_id.duplicated().any():
        raise ValueError("later candidate identity is not unique")
    if later.label_valid.isna().any():
        raise ValueError("later labels did not join for every candidate")

    OUT.mkdir(parents=True)
    prediction_parts: list[pd.DataFrame] = []
    model_contracts: dict[str, object] = {}
    for side in ("long", "short"):
        fields = list(contract[side])
        train = hist.loc[hist.side_name.astype(str).str.lower().eq(side)].copy()
        test = later.loc[later.side_name.astype(str).str.lower().eq(side)].copy()
        missing = [f for f in fields if f not in test.columns]
        if missing:
            raise ValueError(f"later {side} F0 fields missing: {missing}")
        model = _fit(train, fields, seed=17)
        proba = np.asarray(model.predict_proba(_matrix(test, fields)), dtype=np.float32)
        if not np.isfinite(proba).all():
            raise ValueError(f"non-finite {side} F0 probabilities")
        out = test[["candidate_id", "__ts__", "__symbol__", "side_name", "label_valid", "target_invalid", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "robust_clear_event_b25"]].copy()
        out["r3_meta_p_adverse"] = proba[:, 0]
        out["r3_meta_p_weak"] = proba[:, 1]
        out["r3_meta_p_clear"] = proba[:, 2]
        out["base_score"] = out.r3_meta_p_clear - 0.5 * out.r3_meta_p_adverse
        out["label_available_ts"] = pd.to_datetime(test["__label_available_at__"], utc=True).to_numpy()
        prediction_parts.append(out)
        model.booster_.save_model(str(OUT / f"frozen_f0_r3_{side}.model.txt"))
        model_contracts[side] = {"fields": fields, "field_count": len(fields), "params": model.get_params(), "class_order": ["adverse", "weak", "clear"]}

    pred = pd.concat(prediction_parts, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    pred.to_parquet(OUT / "later_frozen_f0_base_predictions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_frozen_f0_later_base_v1", "status": "COMPLETE",
        "historical_panel": str(HISTORICAL), "later_context": str(LATER_CONTEXT),
        "label_contract": "TP6/SL4 H12 exact labels; cost applied once",
        "cost_bps": float(TP6_SL4_COST_BPS), "model_contract": model_contracts,
        "rows": int(len(pred)), "long_rows": int((pred.side_name == "long").sum()), "short_rows": int((pred.side_name == "short").sum()),
        "feature_contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
        "no_target_imputation": True,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return OUT


if __name__ == "__main__":
    print(run())
