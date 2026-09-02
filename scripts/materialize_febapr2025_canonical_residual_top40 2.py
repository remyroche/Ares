#!/usr/bin/env python3
"""Materialise the immutable historical side-local top-40 residual population."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.base_candidate_population import (  # noqa: E402
    BaseCandidatePopulationContract,
    select_base_candidate_population,
)
BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
LABELS = ROOT / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
EXECUTION = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1/labels.parquet"
OUT = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    base = pd.read_parquet(BASE / "oof_predictions.parquet")
    source = base.loc[:, ["candidate_id", "side_name", "__symbol__", "__ts__", "base_oof_score"]].rename(columns={"base_oof_score": "score"})
    selected = select_base_candidate_population(source, BaseCandidatePopulationContract(top_fraction=0.40, score_col="score"))
    selected = selected.rename(columns={"score": "base_oof_score", "base_candidate_rank_timestamp_side": "base_rank_timestamp_side", "base_candidate_rank_pct_timestamp_side": "base_rank_pct_timestamp_side", "base_candidate_group_rows": "base_group_rows"})
    if selected.candidate_id.duplicated().any() or not selected.selected_top40.all():
        raise RuntimeError("top-40 selection identity failure")
    shards = sorted(LABELS.glob("train_global_*_5_2025_0[234].parquet"))
    native = pd.concat([pd.read_parquet(p, columns=["candidate_id", "__first_touch_target_soft__", "__w__", "__first_touch_capture_net__", "__decision_ts__"]) for p in shards], ignore_index=True)
    execution = pd.read_parquet(EXECUTION, columns=["candidate_id", "execution_net_ev_12h", "execution_label_end_utc"])
    result = selected.merge(native, on="candidate_id", how="left", validate="one_to_one").merge(execution, on="candidate_id", how="left", validate="one_to_one")
    if len(result) != len(selected) or result[["__first_touch_target_soft__", "__w__", "__first_touch_capture_net__", "execution_net_ev_12h"]].isna().any().any():
        raise RuntimeError("top-40 native/execution label identity join failed")
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True)
    result["native_label_resolution_utc"] = result["__decision_ts__"] + pd.Timedelta(hours=24)
    result["execution_label_end_utc"] = pd.to_datetime(result["execution_label_end_utc"], utc=True)
    OUT.mkdir(parents=True)
    result.to_parquet(OUT / "population.parquet", index=False, compression="zstd")
    manifest = {"schema": "febapr2025_canonical_residual_top40_v1", "status": "IMMUTABLE_BASE_OOF_TOP40", "base_oof_sha256": _sha(BASE / "oof_predictions.parquet"), "base_rows": int(len(base)), "selected_rows": int(len(result)), "top_fraction": 0.40, "rank_scope": "timestamp_side", "native_target": "__first_touch_capture_net__", "native_weight": "__w__", "execution_economic_diagnostic": "execution_net_ev_12h", "population_sha256": _sha(OUT / "population.parquet")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
