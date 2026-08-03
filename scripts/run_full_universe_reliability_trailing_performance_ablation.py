#!/usr/bin/env python3
"""Causal trailing base-hit performance feature ablation for the reliability head.

Each feature at time t uses only labels whose H12 availability time is strictly
before t.  ``hit`` is the realised TP3/SL2 upper-barrier event and ``surprise``
is ``hit - frozen p_upper``: a calibrated base has trailing surprise near zero.
The model contract is otherwise the frozen P(net>25bps), B2-P30 reliability
head.  Existing 30 context fields are retained and the twelve requested
global/per-asset, 3/5/10-day fields are added as mandatory context inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_round_b_meta_targets import (  # noqa: E402
    _attach_prequential_expected, _attach_prequential_population, _base_predictions,
    _state_features,
)

PERFORMANCE_COLUMNS = tuple(
    f"base_{scope}_{kind}_{days}d"
    for scope in ("global", "asset")
    for kind in ("hit_rate", "hit_surprise")
    for days in (3, 5, 10)
)


def _rolling_at_availability(history: pd.DataFrame, *, grouped: bool) -> pd.DataFrame:
    """Hourly trailing sums, closed-left: availability exactly at t is excluded."""
    history = history.copy()
    history["availability_hour"] = pd.to_datetime(history["__label_available_at__"], utc=True).dt.floor("h")
    keys = (["__symbol__"] if grouped else []) + ["availability_hour"]
    aggregate = history.groupby(keys, observed=True).agg(
        count=("hit", "size"), hit=("hit", "sum"), surprise=("surprise", "sum")
    ).reset_index()
    end = history["__ts__"].max().ceil("h")
    start = history["__ts__"].min().floor("h") - pd.Timedelta(days=10)
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    if not grouped:
        base = aggregate.set_index("availability_hour").reindex(hours, fill_value=0.0)
        out = pd.DataFrame({"__ts__": hours})
        for days in (3, 5, 10):
            values = base[["count", "hit", "surprise"]].rolling(f"{days}D", closed="left").sum()
            count = values["count"].to_numpy(float)
            out[f"base_global_hit_rate_{days}d"] = np.divide(values["hit"].to_numpy(float), count, out=np.full(len(out), np.nan), where=count > 0)
            out[f"base_global_hit_surprise_{days}d"] = np.divide(values["surprise"].to_numpy(float), count, out=np.full(len(out), np.nan), where=count > 0)
        return out
    outputs = []
    for symbol, item in aggregate.groupby("__symbol__", observed=True, sort=False):
        base = item.set_index("availability_hour")[["count", "hit", "surprise"]].reindex(hours, fill_value=0.0)
        out = pd.DataFrame({"__ts__": hours, "__symbol__": symbol})
        for days in (3, 5, 10):
            values = base.rolling(f"{days}D", closed="left").sum()
            count = values["count"].to_numpy(float)
            out[f"base_asset_hit_rate_{days}d"] = np.divide(values["hit"].to_numpy(float), count, out=np.full(len(out), np.nan), where=count > 0)
            out[f"base_asset_hit_surprise_{days}d"] = np.divide(values["surprise"].to_numpy(float), count, out=np.full(len(out), np.nan), where=count > 0)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def _attach_performance(data: pd.DataFrame) -> pd.DataFrame:
    source = data[["__ts__", "__label_available_at__", "__symbol__", "event", "p_upper"]].copy()
    source["hit"] = source["event"].eq(0).astype(float)
    source["surprise"] = source["hit"] - source["p_upper"].to_numpy(float)
    global_features = _rolling_at_availability(source, grouped=False)
    asset_features = _rolling_at_availability(source, grouped=True)
    output = data.merge(global_features, on="__ts__", how="left", validate="many_to_one")
    output = output.merge(asset_features, on=["__ts__", "__symbol__"], how="left", validate="many_to_one")
    if set(PERFORMANCE_COLUMNS) - set(output):
        raise RuntimeError("trailing performance feature join is incomplete")
    return output


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.05, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=10.,
        random_state=20260804, n_jobs=1, verbosity=-1,
    )


def _read(panel: Path, context: list[str]) -> pd.DataFrame:
    cols = ["candidate_id", "__ts__", "__symbol__", "__label_available_at__", "side_name", "t4_tp3_sl2_net_bps", "t4_tp3_sl2_gross_bps", "t2_tp3_sl2_event", *context]
    frames = [pd.read_parquet(path, columns=cols) for path in sorted((panel / "parts").glob("*.parquet"))]
    raw = pd.concat(frames, ignore_index=True).rename(columns={"t4_tp3_sl2_net_bps": "net_bps", "t4_tp3_sl2_gross_bps": "gross_bps", "t2_tp3_sl2_event": "event"})
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True)
    raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
    return raw


def _run_one(data: pd.DataFrame, *, train_start: pd.Timestamp, oos_start: pd.Timestamp, oos_end: pd.Timestamp, base_features: list[str]) -> tuple[pd.DataFrame, dict]:
    mapped = _attach_prequential_expected(data, pd.Timestamp("2024-04-15", tz="UTC"), oos_start)
    mapped = _attach_prequential_population(mapped, .30, train_start, oos_start)
    train = mapped[mapped.__ts__.ge(train_start) & mapped.__ts__.lt(oos_start) & mapped.__label_available_at__.lt(oos_start) & mapped.high_base_eligible].copy()
    evaluation = mapped[mapped.__ts__.ge(oos_start) & mapped.__ts__.lt(oos_end)].copy()
    eligible = evaluation[evaluation.high_base_eligible].copy()
    # Missing values occur only before enough resolved history exists.  The
    # explicit feature is still causal; LightGBM learns the warm-up state.
    model = _model().fit(_state_features(train, base_features), train.net_bps.gt(25.).to_numpy(int))
    score = model.predict_proba(_state_features(eligible, base_features))[:, 1]
    output = evaluation[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_expected_net_bps", "base_expected_gross_bps", "base_payoff_mixture_sd_bps", "high_base_eligible", "high_base_cutoff"]].copy()
    output["reliability_score"] = np.nan
    output.loc[eligible.index, "reliability_score"] = score
    return output, {"train_rows": len(train), "eligible_rows": len(eligible), "features": base_features}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3")
    p.add_argument("--audit", type=Path, default=ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/layer_feature_contract_audit.json")
    p.add_argument("--base-root", type=Path, default=ROOT / "data_perp/artifacts/full_universe_base_hpo_20260802_v1")
    p.add_argument("--baseline-manifest", type=Path, default=ROOT / "data_perp/artifacts/full_universe_round2_reliability_net_gt_25_oos_20260804_v1/manifest.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    context = json.loads(a.audit.read_text())["meta"]["coverage_ge_90pct"]
    base_features = json.loads(a.baseline_manifest.read_text())["meta_features"]
    raw = _read(a.panel, context)
    base = _base_predictions(a.base_root, "tp3_sl2")
    data = raw.merge(base, on="candidate_id", validate="one_to_one")
    data = _attach_performance(data)
    coverage = {c: float(data[c].notna().mean()) for c in PERFORMANCE_COLUMNS}
    augmented = [*base_features, *PERFORMANCE_COLUMNS]
    dev, dev_info = _run_one(data, train_start=pd.Timestamp("2024-05-01", tz="UTC"), oos_start=pd.Timestamp("2024-06-15", tz="UTC"), oos_end=pd.Timestamp("2024-08-01", tz="UTC"), base_features=augmented)
    oos, oos_info = _run_one(data, train_start=pd.Timestamp("2024-05-01", tz="UTC"), oos_start=pd.Timestamp("2024-08-01", tz="UTC"), oos_end=pd.Timestamp("2024-12-01", tz="UTC"), base_features=augmented)
    for source, destination in ((dev, "development_reliability.parquet"), (oos, "oos_reliability.parquet")):
        source.to_parquet(a.out / destination, index=False)
    manifest = {"schema": "full_universe_causal_trailing_base_performance_v1", "target": "I(TP3/SL2 realised net >25bps)", "performance_definition": {"hit": "I(realised TP3/SL2 upper barrier first)", "surprise": "hit - frozen p_upper", "availability": "label_available_at < feature timestamp", "windows_days": [3, 5, 10], "scopes": ["global", "asset"]}, "performance_features": list(PERFORMANCE_COLUMNS), "coverage": coverage, "base_context_features": base_features, "augmented_context_features": augmented, "development": dev_info, "oos": oos_info, "selection_rule": "frozen 75% rank(value)/25% rank(reliability), causal B2 P30 admission; no refit of value/residual"}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"coverage": coverage, "dev": dev_info, "oos": oos_info}, indent=2))


if __name__ == "__main__":
    main()
