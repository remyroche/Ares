#!/usr/bin/env python3
"""Evaluate residual-archetype recognizers on future rows from held-out symbols."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    add_reference_surprise_targets,
    strip_outcomes_for_oos,
)
from extreme_price_movements.meta_residual_surprise_heads import (  # noqa: E402
    ResidualSurpriseHeadState,
)
from scripts.run_meta_residual_ae_representation_ablation import (
    _candidate_features,  # noqa: E402
)
from scripts.run_meta_residual_pca_representation_ablation import (  # noqa: E402
    _fit_pca,
    _transform_pca,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)


def _heldout(symbol: str) -> bool:
    digest = hashlib.sha256(str(symbol).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 5 == 0


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    if valid.sum() < 100 or len(np.unique(y[valid])) < 2:
        return np.nan
    return float(average_precision_score(y[valid], score[valid]))


def _lift(y: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    y = y[valid]
    score = score[valid]
    if len(y) < 100 or float(y.mean()) <= 0.0:
        return np.nan
    keep = max(1, int(np.ceil(fraction * len(score))))
    idx = np.argpartition(score, len(score) - keep)[-keep:]
    return float(y[idx].mean() / y.mean())


def _record(
    name: str, prepared: pd.DataFrame, generated: pd.DataFrame
) -> dict[str, Any]:
    signed = pd.to_numeric(prepared["hit_surprise"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    negative = pd.to_numeric(prepared["negative_tail_label"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    positive = pd.to_numeric(prepared["positive_tail_label"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    expected = pd.to_numeric(
        generated["meta_resid_arch_expected_hit_surprise"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    finite = np.isfinite(signed) & np.isfinite(expected)
    constant = float(np.nanmean(signed))
    mse = float(np.mean((signed[finite] - expected[finite]) ** 2))
    baseline_mse = float(np.mean((signed[finite] - constant) ** 2))
    date = pd.to_datetime(prepared["__ts__"], utc=True).dt.floor("D")
    daily = (
        pd.DataFrame({"date": date, "surprise": signed})
        .groupby("date")["surprise"]
        .mean()
    )
    threshold = float(daily.abs().quantile(0.90))
    return {
        "scope": name,
        "rows": int(len(prepared)),
        "symbols": int(prepared["__symbol__"].nunique()),
        "days": int(date.nunique()),
        "future_tail_event_days": int(daily.abs().ge(threshold).sum()),
        "signed_surprise_mse": mse,
        "constant_baseline_mse": baseline_mse,
        "signed_surprise_correlation": float(
            np.corrcoef(signed[finite], expected[finite])[0, 1]
        ),
        "negative_tail_prevalence": float(np.nanmean(negative)),
        "negative_tail_ap": _ap(negative, -expected),
        "negative_tail_top_decile_lift": _lift(negative, -expected),
        "positive_tail_prevalence": float(np.nanmean(positive)),
        "positive_tail_ap": _ap(positive, expected),
        "positive_tail_top_decile_lift": _lift(positive, expected),
    }


def _head_record(
    name: str,
    prepared: pd.DataFrame,
    labels: pd.DataFrame,
    generated: pd.DataFrame,
) -> dict[str, Any]:
    signed = labels["signed_surprise"].to_numpy(dtype=np.float32)
    negative = labels["negative_tail"].to_numpy(dtype=np.float32)
    positive = labels["positive_tail"].to_numpy(dtype=np.float32)
    signed_pred = generated["meta_resid_signed_surprise_prediction"].to_numpy(
        dtype=np.float32
    )
    negative_pred = generated["meta_resid_negative_tail_probability"].to_numpy(
        dtype=np.float32
    )
    positive_pred = generated["meta_resid_positive_tail_probability"].to_numpy(
        dtype=np.float32
    )
    finite = np.isfinite(signed) & np.isfinite(signed_pred)
    constant = float(np.nanmean(signed))
    date = pd.to_datetime(prepared["__ts__"], utc=True).dt.floor("D")
    daily = (
        pd.DataFrame({"date": date, "surprise": signed})
        .groupby("date")["surprise"]
        .mean()
    )
    threshold = float(daily.abs().quantile(0.90))
    return {
        "scope": name,
        "rows": int(len(prepared)),
        "symbols": int(prepared["__symbol__"].nunique()),
        "days": int(date.nunique()),
        "future_tail_event_days": int(daily.abs().ge(threshold).sum()),
        "signed_surprise_mse": float(
            np.mean((signed[finite] - signed_pred[finite]) ** 2)
        ),
        "constant_baseline_mse": float(np.mean((signed[finite] - constant) ** 2)),
        "signed_surprise_correlation": float(
            np.corrcoef(signed[finite], signed_pred[finite])[0, 1]
        ),
        "negative_tail_prevalence": float(np.nanmean(negative)),
        "negative_tail_ap": _ap(negative, negative_pred),
        "negative_tail_top_decile_lift": _lift(negative, negative_pred),
        "positive_tail_prevalence": float(np.nanmean(positive)),
        "positive_tail_ap": _ap(positive, positive_pred),
        "positive_tail_top_decile_lift": _lift(positive, positive_pred),
    }


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    data = pd.read_parquet(root / "cache" / "compact_reference_with_lifecycle.parquet")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    holdout = data["__symbol__"].astype(str).map(_heldout)
    split = pd.Timestamp("2026-05-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    train = data[data["__ts__"].lt(split) & ~holdout].copy()
    valid = data[data["__ts__"].ge(split) & data["__ts__"].lt(end)].copy()
    candidates = _candidate_features(data, root)
    pca_inputs = candidates[: min(80, len(candidates))]
    pca = _fit_pca(
        train,
        pca_inputs,
        seed=20260711,
        requested_components=8,
        scaled_clip=8.0,
    )
    train_pca = _transform_pca(train, pca)
    valid_pca = _transform_pca(valid, pca)
    for name in train_pca.columns:
        train[name] = train_pca[name].to_numpy(dtype=np.float32, copy=False)
        valid[name] = valid_pca[name].to_numpy(dtype=np.float32, copy=False)
    recognizer = ResidualArchetypeRecognizer(
        ResidualArchetypeConfig(
            use_residual_ae_gmm=False,
            random_state=20260711,
        ),
        list(dict.fromkeys([*candidates, *train_pca.columns.astype(str).tolist()])),
    ).fit(train)
    prepared = add_reference_surprise_targets(valid, recognizer.config)
    generated = recognizer.transform_oos(strip_outcomes_for_oos(valid))
    surprise_head = ResidualSurpriseHeadState(
        candidate_features=list(
            dict.fromkeys([*candidates, *train_pca.columns.astype(str).tolist()])
        ),
        config=ResidualArchetypeConfig(random_state=20260711),
    ).fit(train)
    head_generated = surprise_head.transform(strip_outcomes_for_oos(valid))
    head_labels = surprise_head.labels(valid)
    valid_prepared = add_reference_surprise_targets(valid, recognizer.config)
    top20 = valid_prepared["reference_rank_pct"].ge(0.80).to_numpy(dtype=bool)
    valid_holdout = valid["__symbol__"].astype(str).map(_heldout).to_numpy(dtype=bool)
    rows = [
        _record(
            "future_heldout_symbols",
            prepared.loc[valid_holdout],
            generated.loc[valid_holdout],
        ),
        _record(
            "future_seen_symbols",
            prepared.loc[~valid_holdout],
            generated.loc[~valid_holdout],
        ),
        _record("future_all_symbols", prepared, generated),
        _head_record(
            "surprise_head_future_heldout_symbols_top20",
            valid_prepared.loc[valid_holdout & top20],
            head_labels.loc[valid_holdout & top20],
            head_generated.loc[valid_holdout & top20],
        ),
        _head_record(
            "surprise_head_future_seen_symbols_top20",
            valid_prepared.loc[~valid_holdout & top20],
            head_labels.loc[~valid_holdout & top20],
            head_generated.loc[~valid_holdout & top20],
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(report_dir / "stage12_recognizer_asset_portability.csv", index=False)
    held = metrics[
        metrics["scope"].eq("surprise_head_future_heldout_symbols_top20")
    ].iloc[0]
    manifest = {
        "schema": "meta_residual_recognizer_asset_portability_v1",
        "train_end": str(train["__ts__"].max()),
        "evaluation_start": str(valid["__ts__"].min()),
        "evaluation_end": str(valid["__ts__"].max()),
        "heldout_symbol_fraction": 0.20,
        "heldout_symbols": int(valid.loc[valid_holdout, "__symbol__"].nunique()),
        "heldout_rows": int(valid_holdout.sum()),
        "future_event_days": int(held["future_tail_event_days"]),
        "pca_effective_rank": float(pca["effective_rank"]),
        "surprise_head": surprise_head.manifest(),
        "negative_tail_ap_above_prevalence": bool(
            held["negative_tail_ap"] > held["negative_tail_prevalence"]
        ),
        "positive_tail_ap_above_prevalence": bool(
            held["positive_tail_ap"] > held["positive_tail_prevalence"]
        ),
        "top_decile_lift_pass": bool(
            held["negative_tail_top_decile_lift"] > 1.0
            and held["positive_tail_top_decile_lift"] > 1.0
        ),
        "mse_beats_constant": bool(
            held["signed_surprise_mse"] < held["constant_baseline_mse"]
        ),
        "portability_pass": bool(
            held["negative_tail_ap"] > held["negative_tail_prevalence"]
            and held["positive_tail_ap"] > held["positive_tail_prevalence"]
            and held["negative_tail_top_decile_lift"] > 1.0
            and held["positive_tail_top_decile_lift"] > 1.0
        ),
        "leakage_contract": (
            "PCA and recognizer fit only on non-held-out symbols before May; May-June rows "
            "and every held-out symbol are unseen at fit time."
        ),
    }
    (report_dir / "stage12_recognizer_asset_portability_manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
