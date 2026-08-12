#!/usr/bin/env python3
"""Retrain side-local R3 base models on the unified two-year ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_ledger_20260806_v1/ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_base_oof_20260806_v1"
BASE_SELECTION = ROOT / "data_perp/artifacts/stage_i_base_selection_20260803_v5"
FOLDS = (
    ("fold_00", pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-09-01", tz="UTC")),
    ("fold_01", pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC")),
    ("fold_02", pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-09-01", tz="UTC")),
)
TAILS = (0.01, 0.05, 0.10)


def _features(side: str, ledger_columns: set[str]) -> list[str]:
    manifest = json.loads((BASE_SELECTION / side / "manifest.json").read_text())
    selected = [str(x) for x in manifest["selected_features"] if str(x) in ledger_columns]
    if len(selected) < 10:
        raise ValueError(f"{side}: too few selected base features in the unified ledger")
    return selected


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, (v, w) in enumerate(zip(values, weights)):
        levels.append(float(v)); masses.append(float(w)); starts.append(i); ends.append(i + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            mass = masses[-2] + masses[-1]
            levels[-2] = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / mass
            masses[-2] = mass; ends[-2] = ends[-1]
            levels.pop(); masses.pop(); starts.pop(); ends.pop()
    out = np.empty(len(values), dtype=float)
    for v, left, right in zip(levels, starts, ends):
        out[left:right] = v
    return out


def _fit_score_map(train_score: np.ndarray, train_net: np.ndarray, test_score: np.ndarray, bins: int = 20) -> np.ndarray:
    ok = np.isfinite(train_score) & np.isfinite(train_net)
    if ok.sum() < 50 or not np.isfinite(test_score).any():
        return np.full(len(test_score), np.nan, dtype=np.float32)
    edges = np.unique(np.nanquantile(train_score[ok], np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        lo, hi = float(np.nanmin(train_score[ok])), float(np.nanmax(train_score[ok]))
        hi = max(hi, lo + 1e-6)
        edges = np.linspace(lo, hi, bins + 1)
    n = len(edges) - 1
    b = np.clip(np.searchsorted(edges[1:-1], train_score[ok], side="right"), 0, n - 1)
    count = np.bincount(b, minlength=n).astype(float)
    total = np.bincount(b, weights=train_net[ok], minlength=n).astype(float)
    mean = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    global_mean = float(np.nanmean(train_net[ok]))
    mean = np.where(np.isfinite(mean), mean, global_mean)
    fitted = _pava(mean, np.maximum(count, 1.0))
    out = np.full(len(test_score), np.nan, dtype=np.float32)
    valid = np.isfinite(test_score)
    tb = np.clip(np.searchsorted(edges[1:-1], test_score[valid], side="right"), 0, n - 1)
    out[valid] = fitted[tb].astype(np.float32)
    return out


def _metrics(frame: pd.DataFrame, score_col: str, fold: str, scope: str, key: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tail in TAILS:
        n = max(1, int(np.ceil(len(frame) * tail)))
        top = frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
        rows.append({"fold": fold, "scope": scope, "key": key, "tail": tail, "rows": len(top), "pool_rows": len(frame), "net_bps": float(top.exact_net_bps.mean()), "gross_bps": float(top.exact_gross_bps.mean()), "rank_ic": float(frame[score_col].rank().corr(frame.exact_net_bps.rank()))})
    return rows


def run(ledger_path: Path = DEFAULT_LEDGER, out: Path = DEFAULT_OUT) -> Path:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Read only the declared identity/labels plus side-specific feature union.
    import pyarrow.parquet as pq
    columns = set(pq.ParquetFile(ledger_path).schema.names)
    feature_map = {side: _features(side, columns) for side in ("long", "short")}
    keep = ["candidate_id", "__ts__", "side_name", "label_available_ts", "decision_ts", "exact_gross_bps", "exact_net_bps", "r3_class"]
    keep += sorted(set(feature_map["long"]) | set(feature_map["short"]))
    d = pd.read_parquet(ledger_path, columns=keep)
    d["__ts__"] = pd.to_datetime(d["__ts__"], utc=True); d["label_available_ts"] = pd.to_datetime(d["label_available_ts"], utc=True); d["decision_ts"] = pd.to_datetime(d["decision_ts"], utc=True)
    d = d.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for fold, start, end in FOLDS:
        test_mask = (d.__ts__ >= start) & (d.__ts__ < end)
        test_all = d.loc[test_mask].copy()
        train_all = d.loc[(d.__ts__ < start) & (d.label_available_ts < start)].copy()
        if len(train_all) < 1000 or test_all.empty:
            continue
        for side in ("long", "short"):
            train = train_all[train_all.side_name.eq(side)].copy(); test = test_all[test_all.side_name.eq(side)].copy()
            fields = feature_map[side]
            X = train[fields].replace([np.inf, -np.inf], np.nan); Xt = test[fields].replace([np.inf, -np.inf], np.nan)
            med = X.median().fillna(0.0); X = X.fillna(med).astype("float32"); Xt = Xt.fillna(med).astype("float32")
            y = pd.to_numeric(train.r3_class, errors="coerce").to_numpy(int)
            if set(np.unique(y)) - {0, 1, 2}:
                raise ValueError(f"{side}/{fold}: invalid R3 classes")
            model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=220, learning_rate=0.035, num_leaves=24, max_depth=5, min_child_samples=max(80, int(0.01 * len(train))), colsample_bytree=0.85, reg_lambda=20.0, random_state=20260806, n_jobs=1, verbosity=-1)
            model.fit(X, y)
            raw = np.asarray(model.predict_proba(Xt), dtype=float)
            probs = np.zeros((len(test), 3), dtype=float); probs[:, model.classes_.astype(int)] = raw
            score_train = model.predict_proba(X)[:, list(model.classes_).index(2)] - 0.5 * model.predict_proba(X)[:, list(model.classes_).index(0)]
            score_test = probs[:, list(model.classes_).index(2)] - 0.5 * probs[:, list(model.classes_).index(0)]
            mapped = _fit_score_map(score_train, train.exact_net_bps.to_numpy(float), score_test)
            z = test[["candidate_id", "__ts__", "side_name", "exact_gross_bps", "exact_net_bps"]].copy()
            z["fold"] = fold; z["base_p_adverse"] = probs[:, 0].astype("float32"); z["base_p_weak"] = probs[:, 1].astype("float32"); z["base_p_clear"] = probs[:, 2].astype("float32"); z["base_score_raw"] = score_test.astype("float32"); z["base_expected_net_bps"] = mapped
            predictions.append(z)
            metrics.extend(_metrics(z, "base_expected_net_bps", fold, "global", "all"))
            metrics.extend(_metrics(z, "base_expected_net_bps", fold, "side", side))
            for month, m in z.groupby(z.__ts__.dt.strftime("%Y-%m"), sort=True):
                metrics.extend(_metrics(m, "base_expected_net_bps", fold, "month_side", f"{side}:{month}"))
            provenance.append({"fold": fold, "side": side, "train_start": str(train.__ts__.min()), "train_end": str(train.__ts__.max()), "train_rows": len(train), "test_start": str(test.__ts__.min()), "test_end": str(test.__ts__.max()), "test_rows": len(test), "feature_count": len(fields), "features": fields})
    pred = pd.concat(predictions, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    pred.to_parquet(out / "base_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metrics).to_parquet(out / "base_metrics.parquet", index=False)
    pd.DataFrame(provenance).to_json(out / "base_fold_provenance.json", orient="records", indent=2)
    manifest = {"schema": "tp6_sl4_h12_two_year_base_oof_v1", "status": "COMPLETED", "ledger": str(ledger_path), "folds": [{"name": n, "test_start": str(a), "test_end_exclusive": str(b)} for n, a, b in FOLDS], "target": "R3 class from authoritative TP6/SL4/H12 labels", "score": "P(clear) - 0.5 P(adverse)", "mapping": "training-fold side-local monotone score-to-exact-net map", "rows": int(len(pred)), "features_by_side": {k: len(v) for k, v in feature_map.items()}, "outputs": ["base_oof_predictions.parquet", "base_metrics.parquet", "base_fold_provenance.json"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER); parser.add_argument("--out", type=Path, default=DEFAULT_OUT); args = parser.parse_args(); print(run(args.ledger, args.out))
