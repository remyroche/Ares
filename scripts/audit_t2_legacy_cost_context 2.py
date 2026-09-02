#!/usr/bin/env python3
"""Quantify the historical realised-cost-context contamination in T2."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry, soft_event_targets, top_book_metrics
from scripts.run_t2_atr_sequential_funnel import _add_causal_context, _conditional_mean, _huber, _resolved_before


def _score(train: pd.DataFrame, test: pd.DataFrame, raw: list[str], use_legacy_cost: bool) -> np.ndarray:
    cols = [*raw, "side_is_long"] + (["legacy_realised_cost_bps"] if use_legacy_cost else [])
    x, z = train[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)
    target = train[["t2_upper_soft", "t2_lower_soft", "t2_timeout_soft"]].to_numpy(float)
    p = np.column_stack([np.maximum(_huber(x, target[:, k], z), 0.0) for k in range(3)])
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
    means = np.asarray([_conditional_mean(target[:, k], train.execution_net_ev_12h.to_numpy(float)) for k in range(3)]) * 10_000.0
    return p @ means


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--features-json", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    raw = list(validate_feature_columns(json.loads(a.features_json.read_text())["raw_feature_columns"]))
    cols = ["candidate_id", "__decision_ts__", "__label_available_at__", "side_name", "oof_fold", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", *raw]
    x = _add_causal_context(pd.read_parquet(a.ledger, columns=cols))
    events = pd.read_parquet(a.events)
    x = x.merge(events, on="candidate_id", validate="one_to_one")
    x[["t2_upper_soft", "t2_lower_soft", "t2_timeout_soft"]] = soft_event_targets(x, BarrierGeometry(2.0, 1.0), temperature_atr=.25)
    x["legacy_realised_cost_bps"] = x.execution_cost_return.astype(float) * 10_000.0
    base = x.loc[x.oof_fold.eq("base_train")].copy()
    dev = x.loc[x.oof_fold.eq("meta_train")].copy()
    train = _resolved_before(base, dev)
    rows = []
    for name, flag in (("strict_no_cost", False), ("strict_legacy_realised_cost", True)):
        score = _score(train, dev, raw, flag)
        book = dev.loc[:, ["candidate_id", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]].copy()
        book["score_bps"] = score
        metrics = top_book_metrics(book, score_column="score_bps")
        metrics["arm"] = name
        rows.append(metrics)
    stage = Path(tempfile.mkdtemp(prefix=f".{a.output.name}.", dir=a.output.parent))
    try:
        pd.concat(rows, ignore_index=True).to_parquet(stage / "cost_context_ablation.parquet", index=False)
        (stage / "manifest.json").write_text(json.dumps({"purpose": "demonstrate the effect of adding execution_cost_return from the realised target ledger", "strict_label_resolution_purge": True, "geometry": "TP2_SL1", "temperature": .25, "decision": "legacy realised cost is prohibited from T2 inference inputs regardless of result"}, indent=2) + "\n")
        os.replace(stage, a.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
