#!/usr/bin/env python3
"""Aggregate immutable monthly canonical-base OOF shards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _economics(frame: pd.DataFrame) -> dict[str, float | int]:
    ordered = frame.sort_values("base_oof_score", ascending=False, kind="stable")
    k = max(1, int(len(ordered) * 0.10 + 0.999999))
    top = ordered.head(k)
    return {
        "rows": int(len(frame)),
        "top10_global_rows": int(k),
        "mean_execution_net_ev": float(frame.execution_net_ev_12h.mean()),
        "top10_global_execution_net_ev": float(top.execution_net_ev_12h.mean()),
        "top10_global_positive_fraction": float((top.execution_net_ev_12h > 0).mean()),
        "score_target_spearman": float(frame[["base_oof_score", "__first_touch_target_soft__"]].corr(method="spearman").iloc[0, 1]),
    }


def main() -> None:
    shard_roots = [BASE / "shards" / f"{side}_{month}" for side in ("long", "short") for month in ("2025_02", "2025_03", "2025_04")]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in shard_roots]
    predictions = pd.concat([pd.read_parquet(root / "oof_predictions.parquet") for root in shard_roots], ignore_index=True)
    provenance = pd.concat([pd.read_parquet(root / "fold_provenance.parquet") for root in shard_roots], ignore_index=True)
    if predictions.candidate_id.duplicated().any() or len(predictions) != 509868:
        raise RuntimeError("monthly shards do not reproduce the accepted frozen identity count")
    predictions.to_parquet(BASE / "oof_predictions.parquet", index=False, compression="zstd")
    provenance.to_parquet(BASE / "fold_provenance.parquet", index=False, compression="zstd")
    gate = {
        "schema": "febapr2025_canonical_base_oof_gate_v1",
        "status": "PASS_BASE_IDENTITY_AND_OOF_GATE",
        "prediction_rows": int(len(predictions)),
        "folds": int(len(provenance)),
        "per_side": {side: _economics(predictions.loc[predictions.side_name.eq(side)]) for side in ("long", "short")},
        "global": _economics(predictions),
        "latest_month": _economics(predictions.loc[predictions.__ts__.dt.month.eq(4)]),
        "coverage_policy": "label-complete outer OOF rows with LightGBM native missing values; per-feature fold coverage retained in shard gates",
        "scope": "canonical base only; residual and downstream execution heads are intentionally excluded",
        "shards": [{"path": str(root.relative_to(ROOT)), "manifest_sha256": _sha(root / "manifest.json")} for root in shard_roots],
    }
    (BASE / "coverage_economics_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    (BASE / "manifest.json").write_text(json.dumps({"schema": "febapr2025_canonical_base_oof_v1", "status": "AGGREGATED_BASE_ONLY_STRICT_MONTHLY_OOF", "outputs": {name: _sha(BASE / name) for name in ("oof_predictions.parquet", "fold_provenance.parquet", "coverage_economics_gate.json")}, "shard_manifests": [str(root / "manifest.json") for root in shard_roots]}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
