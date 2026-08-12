#!/usr/bin/env python3
"""Materialize strict-OOS canonical head health for 2026 Jan--Jul 10."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import _map_base, _pct, _rank_fit  # noqa: E402

INPUT = ROOT / "data_perp/artifacts/r3_tp6_sl4_meta_target_ablation_20260803_v1/r3_meta_target_oof_predictions.parquet"
MANIFEST = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/run_manifest.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_head_health_2026_v1"
SEED = 20260807
TARGET_MONTHS = tuple(f"2026-{m:02d}" for m in range(1, 8))


def run(*, output_dir: Path = DEFAULT_OUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads(MANIFEST.read_text())
    context = list(manifest["context_features"])
    x = pd.read_parquet(INPUT)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["label_available_ts"] = pd.to_datetime(x["label_available_ts"], utc=True)
    x["month"] = x["__ts__"].dt.strftime("%Y-%m")
    x["base_score"] = pd.to_numeric(x["r3_meta_p_clear"], errors="coerce") - 0.5 * pd.to_numeric(x["r3_meta_p_adverse"], errors="coerce")
    x = x.loc[x.label_valid.fillna(False) & x.exact_net_bps.notna() & x.month.isin([f"2024-{m:02d}" for m in range(2, 13)] + [f"2025-{m:02d}" for m in range(1, 13)] + list(TARGET_MONTHS))].copy()
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in TARGET_MONTHS:
        held = x.loc[x.month.eq(month)].copy()
        train = x.loc[x.__ts__.lt(pd.Timestamp(month, tz="UTC")) & x.label_available_ts.lt(pd.Timestamp(month, tz="UTC"))].copy()
        if held.empty or train.empty:
            continue
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy(); te = held.loc[held.side_name.eq(side)].copy()
            if len(tr) < 500 or te.empty:
                continue
            tr_anchor, te_anchor = _map_base(tr, te)
            tr["base_anchor"] = tr_anchor; te["base_anchor"] = te_anchor
            residual = tr.exact_net_bps.to_numpy(float) - tr_anchor
            grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
            held_out = te[["candidate_id", "__ts__", "side_name", "month", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "base_score", "exact_net_bps", "exact_gross_bps", "label_available_ts", *context]].copy()
            ranks: list[np.ndarray] = []; raws: list[np.ndarray] = []
            for cap in (25, 40, 60, min(73, len(context))):
                fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context[:cap]]
                for equal_month in (False, True):
                    raw_tr, raw_te = _rank_fit(tr, te, fields, grade, equal_month=equal_month, seed=SEED + int(month[-2:]) * 100 + cap + int(equal_month))
                    ranks.append(_pct(raw_te, raw_tr)); raws.append(raw_te)
                    audits.append({"month": month, "side": side, "cap": cap, "equal_month": bool(equal_month), "train_rows": int(len(tr)), "held_rows": int(len(te))})
            r = np.column_stack(ranks); raw = np.column_stack(raws)
            held_out["base_rank"] = _pct(te.base_score.to_numpy(float), tr.base_score.to_numpy(float))
            held_out["consensus_rank"] = np.median(r, axis=1)
            held_out["base_plus_consensus25"] = 0.75 * held_out.base_rank.to_numpy(float) + 0.25 * held_out.consensus_rank.to_numpy(float)
            held_out["consensus_head_rank_std"] = r.std(axis=1)
            held_out["consensus_head_rank_mad"] = np.median(np.abs(r - np.median(r, axis=1, keepdims=True)), axis=1)
            held_out["consensus_head_rank_iqr"] = np.percentile(r, 75, axis=1) - np.percentile(r, 25, axis=1)
            held_out["consensus_head_rank_min"] = r.min(axis=1)
            held_out["consensus_head_rank_max"] = r.max(axis=1)
            held_out["consensus_head_raw_std"] = raw.std(axis=1)
            held_out["consensus_head_agreement_fraction"] = np.mean(np.abs(r - np.median(r, axis=1, keepdims=True)) <= .10, axis=1)
            held_out["base_consensus_disagreement"] = held_out.base_rank.to_numpy(float) - held_out.consensus_rank.to_numpy(float)
            held_out["fold_month"] = month
            parts.append(held_out)
    out = pd.concat(parts, ignore_index=True)
    output_dir.mkdir(parents=True)
    out.to_parquet(output_dir / "canonical_head_health_2026.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(output_dir / "head_fit_audit.parquet", index=False)
    run_manifest = {"schema":"tp6_sl4_canonical_head_health_2026_v1","status":"COMPLETE","target_months":list(TARGET_MONTHS),"rows":int(len(out)),"context_features":context,"strict_oof":True,"base_contract":"frozen R3 p_clear - 0.5 p_adverse; canonical 8-head Base+Consensus","artifacts":["canonical_head_health_2026.parquet","head_fit_audit.parquet","run_manifest.json"]}
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest,indent=2)+"\n")
    return output_dir


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); args=parser.parse_args(); print(run(output_dir=args.output_dir))
