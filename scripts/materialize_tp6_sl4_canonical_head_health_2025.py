#!/usr/bin/env python3
"""Materialize strict-OOF consensus-head health fields for the canonical stack.

The canonical Base+Consensus replay historically persisted only the median
consensus rank.  This companion replay uses the identical frozen head grid,
queries, labels, fields, seeds, and monthly folds, but retains each head's
OOF raw/rank output plus dispersion/agreement summaries.  It is intentionally
limited to the 2025 development population and never uses held-month labels in
the predictions for that month.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import (
    MONTHS,
    SEED,
    _group,
    _load,
    _map_base,
    _pct,
    _rank_fit,
)

DEFAULT_BASE = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_head_health_2025_v1"


def _head_name(cap: int, equal_month: bool) -> str:
    return f"h_cap{cap}_{'equal_month' if equal_month else 'ordinary'}"


def run(*, base_predictions: Path = DEFAULT_BASE, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    panel, context, context_hash = _load()
    base = pd.read_parquet(base_predictions)
    base["__ts__"] = pd.to_datetime(base["__ts__"], utc=True)
    panel = panel.merge(
        base[["candidate_id", "base_plus_consensus25", "base_rank", "consensus_rank"]],
        on="candidate_id", how="left", validate="one_to_one", suffixes=("", "_base"),
    )
    # The source panel already contains these fields; avoid accidental suffix
    # ambiguity and ensure the canonical score is the persisted one.
    for col in ("base_plus_consensus25", "base_rank", "consensus_rank"):
        if f"{col}_base" in panel:
            panel[col] = panel[f"{col}_base"]
            panel.drop(columns=[f"{col}_base"], inplace=True)

    # Keep the preceding available history so January 2025 can be scored with
    # the same causal schedule.  The persisted canonical scores are available
    # only for 2025; earlier rows receive the exact same score construction
    # from the OOF head ranks generated below.
    months = sorted(set(panel.month.astype(str)) & set([f"2024-{m:02d}" for m in range(2, 13)] + list(MONTHS)))
    caps = (25, 40, 60, min(73, len(context)))
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in months:
        held = panel.loc[panel.month.eq(month)].copy()
        train = panel.loc[
            (panel.__ts__ < pd.Timestamp(month, tz="UTC"))
            & (panel.label_available_ts < pd.Timestamp(month, tz="UTC"))
        ].copy()
        if held.empty or train.empty:
            continue
        for side in ("long", "short"):
            tr = train.loc[train.side_name.eq(side)].copy()
            te = held.loc[held.side_name.eq(side)].copy()
            if len(tr) < 300 or te.empty:
                continue
            tr_anchor, te_anchor = _map_base(tr, te)
            tr["base_anchor"] = tr_anchor
            te["base_anchor"] = te_anchor
            residual = tr.exact_net_bps.to_numpy(float) - tr.base_anchor.to_numpy(float)
            grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
            held_out = te[["candidate_id", "__ts__", "side_name", "month"]].copy()
            head_rank_cols: list[str] = []
            head_raw_cols: list[str] = []
            for cap in caps:
                fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context[:cap]]
                for equal_month in (False, True):
                    name = _head_name(cap, equal_month)
                    raw_tr, raw_te = _rank_fit(
                        tr, te, fields, grade, equal_month=equal_month,
                        seed=SEED + int(month[-2:]) * 100 + cap + int(equal_month),
                    )
                    rank_tr = _pct(raw_tr, raw_tr)
                    rank_te = _pct(raw_te, raw_tr)
                    held_out[f"{name}__raw"] = raw_te
                    held_out[f"{name}__rank"] = rank_te
                    head_rank_cols.append(f"{name}__rank")
                    head_raw_cols.append(f"{name}__raw")
                    audits.append({
                        "month": month, "side": side, "head": name,
                        "cap": cap, "equal_month": bool(equal_month),
                        "train_rows": int(len(tr)), "held_rows": int(len(te)),
                        "query_groups": int(_group(tr)[1].size), "fields": int(len(fields)),
                    })
            ranks = held_out[head_rank_cols].to_numpy(float)
            raws = held_out[head_raw_cols].to_numpy(float)
            held_out["consensus_head_rank_mean"] = np.nanmean(ranks, axis=1)
            held_out["consensus_head_rank_median"] = np.nanmedian(ranks, axis=1)
            held_out["consensus_head_rank_std"] = np.nanstd(ranks, axis=1)
            held_out["consensus_head_rank_mad"] = np.nanmedian(np.abs(ranks - np.nanmedian(ranks, axis=1, keepdims=True)), axis=1)
            held_out["consensus_head_rank_iqr"] = np.nanpercentile(ranks, 75, axis=1) - np.nanpercentile(ranks, 25, axis=1)
            held_out["consensus_head_rank_min"] = np.nanmin(ranks, axis=1)
            held_out["consensus_head_rank_max"] = np.nanmax(ranks, axis=1)
            held_out["consensus_head_raw_std"] = np.nanstd(raws, axis=1)
            med = held_out["consensus_head_rank_median"].to_numpy(float)[:, None]
            held_out["consensus_head_agreement_fraction"] = np.nanmean(np.abs(ranks - med) <= 0.10, axis=1)
            # Add the exact canonical rank components from the persisted panel
            # where available; otherwise construct them from this fold's OOF
            # base/head ranks for the historical training substrate.
            if te["base_plus_consensus25"].notna().all() if "base_plus_consensus25" in te else False:
                source = te[["candidate_id", "base_rank", "consensus_rank", "base_plus_consensus25", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "base_score", "exact_net_bps", "exact_gross_bps", "label_available_ts"]]
            else:
                base_rank_te = _pct(te.base_score.to_numpy(float), tr.base_score.to_numpy(float))
                consensus_rank_te = np.nanmedian(ranks, axis=1).astype(np.float32)
                source = te[["candidate_id", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "base_score", "exact_net_bps", "exact_gross_bps", "label_available_ts"]].copy()
                source["base_rank"] = base_rank_te
                source["consensus_rank"] = consensus_rank_te
                source["base_plus_consensus25"] = 0.75 * base_rank_te + 0.25 * consensus_rank_te
            held_out = held_out.drop(columns=["base_plus_consensus25"], errors="ignore").merge(source, on="candidate_id", how="left", validate="one_to_one")
            held_out["base_consensus_disagreement"] = held_out["base_rank"].to_numpy(float) - held_out["consensus_rank"].to_numpy(float)
            held_out["fold_month"] = month
            parts.append(held_out)
    out = pd.concat(parts, ignore_index=True)
    # Match the canonical development panel exactly.
    expected = base.loc[base.side_name.isin(["long", "short"]), ["candidate_id"]].drop_duplicates()
    # Keep the fold population audit explicit; this catches source rows that
    # the frozen panel cannot score rather than silently dropping them.
    print(json.dumps({"oof_rows": int(len(out)), "expected_rows": int(len(expected)), "oof_months": out.fold_month.value_counts().sort_index().to_dict()}, default=int), flush=True)
    if not set(expected.candidate_id.astype(str)).issubset(set(out.candidate_id.astype(str))):
        missing = sorted(set(expected.candidate_id.astype(str)) - set(out.candidate_id.astype(str)))[:10]
        extra = sorted(set(out.candidate_id.astype(str)) - set(expected.candidate_id.astype(str)))[:10]
        raise RuntimeError(f"OOF candidate mismatch; missing={missing}, extra={extra}")
    output_dir.mkdir(parents=True)
    out.to_parquet(output_dir / "canonical_head_health_2025.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(output_dir / "head_fit_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_canonical_head_health_2025_v1",
        "status": "COMPLETE",
        "source_base_predictions": str(base_predictions),
        "rows": int(len(out)), "heads": 8, "caps": list(caps),
        "target": "canonical TP6/SL4 residual grades [-150,-50,+50,+150]",
        "query": "4-hour UTC x side",
        "strict_oof": True,
        "context_fields": context, "context_sha256": context_hash,
        "artifacts": ["canonical_head_health_2025.parquet", "head_fit_audit.parquet", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(base_predictions=args.base_predictions, output_dir=args.output_dir))
