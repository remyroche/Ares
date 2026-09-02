#!/usr/bin/env python3
"""Fit strictly-OOS directional OI-positioning heads and export entry features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/causal_oi_positioning_2025_train_2026_score_20260831_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_oi_positioning_heads_oof_20260831_v1"
SEED = 1729
FEATURES = (
    "zone_distance_atr", "zone_age_hours", "formation_strength", "build_count", "oi_persistence_ratio",
    "oi_persistent", "oi_unwinding", "qualified_revisit_count", "historical_defended_rate",
)
TARGETS = ("y_defended", "y_failure", "y_trap", "y_unwind")
OUTPUT_FIELDS = (
    "oi_long_build_support_probability", "oi_short_build_resistance_probability",
    "oi_failure_probability_long_build", "oi_failure_probability_short_build",
    "oi_trap_probability_long_build", "oi_trap_probability_short_build",
    "oi_unwind_probability_long_build", "oi_unwind_probability_short_build",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, FEATURES].copy()
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=260, learning_rate=.03, max_depth=3, num_leaves=7,
        min_child_samples=140, subsample=.80, colsample_bytree=.85, reg_lambda=14.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def _snapshot_rows(snapshots: pd.DataFrame) -> pd.DataFrame:
    keys = ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]
    frames: list[pd.DataFrame] = []
    for kind in ("long_build", "short_build"):
        prefix = f"oi_{kind}"
        available = snapshots.get(f"{prefix}_available", pd.Series(False, index=snapshots.index)).fillna(False)
        data = snapshots.loc[available, keys].copy()
        if data.empty:
            continue
        data["kind"] = kind
        for feature in FEATURES:
            source = f"{prefix}__{feature}"
            data[feature] = pd.to_numeric(snapshots.loc[available, source], errors="coerce") if source in snapshots else np.nan
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*keys, "kind", *FEATURES])


def _wide(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]
    prediction_columns = [f"oi_{target.removeprefix('y_')}_probability" for target in TARGETS]
    wide = rows.pivot_table(index=keys, columns="kind", values=prediction_columns, aggfunc="first").reset_index()
    wide.columns = ["_".join(part for part in value if part) if isinstance(value, tuple) else value for value in wide.columns]
    rename: dict[str, str] = {}
    for column in prediction_columns:
        for kind in ("long_build", "short_build"):
            source = f"{column}_{kind}"
            if source not in wide:
                continue
            if kind == "long_build" and column == "oi_defended_probability":
                rename[source] = "oi_long_build_support_probability"
            elif kind == "short_build" and column == "oi_defended_probability":
                rename[source] = "oi_short_build_resistance_probability"
            else:
                rename[source] = source
    wide = wide.rename(columns=rename)
    for field in OUTPUT_FIELDS:
        if field not in wide:
            wide[field] = np.nan
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held-month", action="append", default=["2026-06", "2026-07", "2026-08"])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    events = pd.read_parquet(source / "positioning_interactions.parquet")
    snapshots = pd.read_parquet(source / "positioning_snapshots.parquet")
    events["event_ts"] = pd.to_datetime(events.event_ts, utc=True, errors="raise")
    events["label_available_ts"] = pd.to_datetime(events.label_available_ts, utc=True, errors="raise")
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots.snapshot_ts, utc=True, errors="raise")
    for column in (*FEATURES, *TARGETS):
        if column not in events:
            raise AssertionError(f"positioning interaction contract missing {column}")
    rows = _snapshot_rows(snapshots)
    output.mkdir(parents=True, exist_ok=False)
    scored_events: list[pd.DataFrame] = []
    scored_snapshots: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for held_raw in args.held_month:
        held = pd.Timestamp(f"{held_raw}-01", tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        train_all = events.loc[events.event_ts.lt(held) & events.label_available_ts.lt(held)].copy()
        event_test = events.loc[events.event_ts.ge(held) & events.event_ts.lt(end) & events.label_available_ts.lt(end)].copy()
        snapshot_test = rows.loc[rows.snapshot_ts.ge(held) & rows.snapshot_ts.lt(end)].copy()
        if len(train_all) < 2_000 or event_test.empty or snapshot_test.empty:
            raise RuntimeError(f"insufficient strictly prior positioning support {held:%Y-%m}: {len(train_all)=}, {len(event_test)=}, {len(snapshot_test)=}")
        for kind in ("long_build", "short_build"):
            train = train_all.loc[train_all.kind.eq(kind)].copy()
            et = event_test.loc[event_test.kind.eq(kind)].copy()
            st = snapshot_test.loc[snapshot_test.kind.eq(kind)].copy()
            if len(train) < 500 or et.empty or st.empty:
                continue
            x_train, x_event, x_snapshot = _features(train), _features(et), _features(st)
            for target in TARGETS:
                y = pd.to_numeric(train[target], errors="raise").astype(int)
                field = f"oi_{target.removeprefix('y_')}_probability"
                et[field] = np.nan
                st[field] = np.nan
                if y.nunique() < 2:
                    continue
                model = _model().fit(x_train, y)
                et[field] = model.predict_proba(x_event)[:, 1]
                st[field] = model.predict_proba(x_snapshot)[:, 1]
                truth = pd.to_numeric(et[target], errors="raise").astype(int)
                pred = pd.to_numeric(et[field], errors="raise")
                metrics.append({
                    "held_month": held.strftime("%Y-%m"), "kind": kind, "head": field, "rows": len(et),
                    "base_rate": float(truth.mean()),
                    "auc": float(roc_auc_score(truth, pred)) if truth.nunique() > 1 else np.nan,
                    "brier": float(brier_score_loss(truth, pred)),
                    "spearman": float(truth.corr(pred, method="spearman")),
                })
            scored_events.append(et)
            scored_snapshots.append(st)
            folds.append({"held_month": held.strftime("%Y-%m"), "kind": kind, "train_rows": len(train), "event_test_rows": len(et), "snapshot_test_rows": len(st), "train_label_max": str(train.label_available_ts.max())})
    event_result = pd.concat(scored_events, ignore_index=True)
    snapshot_result = pd.concat(scored_snapshots, ignore_index=True)
    wide = _wide(snapshot_result)
    event_result.to_parquet(output / "positioning_head_oof_predictions.parquet", index=False, compression="zstd")
    snapshot_result.to_parquet(output / "positioning_snapshot_head_oof_predictions.parquet", index=False, compression="zstd")
    wide.loc[wide.target_kind.eq("entry")].to_parquet(output / "entry_oi_positioning_oof_features.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(output / "head_metrics_by_month.parquet", index=False)
    pd.DataFrame(folds).to_parquet(output / "fold_trace.parquet", index=False)
    manifest = {
        "schema": "causal-oi-positioning-heads-oof-v1", "scope": "offline only; no live mutation",
        "source": str(source), "source_manifest_sha256": _sha256(source / "run_manifest.json"), "folds": folds,
        "heads": {target: {"target": target, "features": list(FEATURES), "model": "LGBM binary depth3 leaves7"} for target in TARGETS},
        "causality": "each held month trains only on resolved labels prior to its start; snapshots use strictly earlier OI observations",
        "seed": SEED,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
