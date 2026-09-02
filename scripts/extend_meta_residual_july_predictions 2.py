#!/usr/bin/env python3
"""Extend frozen July residual-meta predictions from the production candidate ledger.

The production ledger is the authoritative top-30 candidate handoff after the
research ledger ends.  Base/meta models, residual bundle, AE/GMM state, feature
selection, HPO parameters, and rank references remain frozen through June.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.alternative_meta_residual_bundle import (
    AlternativeMetaResidualBundle,
)
from extreme_price_movements.data_store import read_symbol_features

ROOT = Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1")
DEFAULT_EXISTING = (
    ROOT
    / "july_current_contract_policy_comparison"
    / "july_oos_old_new_aligned_predictions.parquet"
)
DEFAULT_BUNDLE = (
    ROOT
    / "july_current_contract_refit"
    / "alternative_meta_residual_current_contract.joblib"
)
DEFAULT_LEDGER = Path(
    "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/"
    "s59_s52_frozen_native_shadow_20260709/prediction_ledger.parquet"
)
DEFAULT_OUTPUT = ROOT / "july_predictions_through_20260711"
JSON_FEATURE_COLUMNS = (
    "base_model_feature_values_json",
    "meta_model_feature_values_json",
)
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_features(raw: Any, requested: set[str]) -> dict[str, float]:
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    if not isinstance(values, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in values.items():
        name = str(name)
        if name not in requested:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out[name] = number
    return out


def _canonical_archetype(value: Any, side: str) -> str:
    text = str(value or "").strip()
    prefix = f"{side}__"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _load_live_candidates(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    requested: Iterable[str],
) -> pd.DataFrame:
    ledger = pd.read_parquet(path)
    ledger["__ts__"] = pd.to_datetime(
        ledger.get("signal_bar_ts", ledger.get("feature_source_max_ts")),
        utc=True,
        errors="coerce",
    )
    ledger["decision_ts"] = pd.to_datetime(
        ledger.get("decision_ts", ledger.get("timestamp")), utc=True, errors="coerce"
    )
    ledger["side_name"] = ledger["side"].astype(str).str.lower()
    ledger["__symbol__"] = ledger["symbol"].astype(str)
    ledger["archetype_policy_key"] = [
        _canonical_archetype(value, side)
        for value, side in zip(ledger["policy_archetype"], ledger["side_name"])
    ]
    ledger = ledger.loc[ledger["__ts__"].ge(start) & ledger["__ts__"].lt(end)].copy()
    ledger = ledger.dropna(subset=["__ts__", "__symbol__", "side_name"])
    ledger = ledger.sort_values("decision_ts", kind="stable").drop_duplicates(
        KEYS, keep="last"
    )

    requested_set = set(str(name) for name in requested)
    records: list[dict[str, float]] = []
    for row in ledger.loc[:, list(JSON_FEATURE_COLUMNS)].itertuples(
        index=False, name=None
    ):
        values: dict[str, float] = {}
        for raw in row:
            values.update(_json_features(raw, requested_set))
        records.append(values)
    unpacked = pd.DataFrame.from_records(records, index=ledger.index)
    if not unpacked.empty:
        unpacked = unpacked.astype(np.float32, copy=False)
    for name in unpacked.columns:
        if name not in ledger.columns:
            ledger[name] = unpacked[name]

    direct_map = {
        "score_meta_base_soft_label": "meta_pred",
        "score_current_reference": "meta_pred",
        "base_score": "base_pred",
        "production_policy_rank": "policy_rank_pct",
        "production_threshold_rank": "threshold_basis_rank_score",
        "production_adjusted_rank": "adjusted_rank_score",
    }
    for target, source in direct_map.items():
        if source in ledger.columns:
            ledger[target] = pd.to_numeric(ledger[source], errors="coerce").astype(
                np.float32
            )
    return ledger.reset_index(drop=True)


def _append_feature_store(
    frame: pd.DataFrame,
    *,
    feature_root: Path,
    requested: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    names = [str(name) for name in requested if str(name) not in frame.columns]
    if not names:
        return frame, {}
    values = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    for symbol, positions_raw in frame.groupby(
        "__symbol__", sort=False
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        path = feature_root / f"symbol={str(symbol).replace('/', '_')}.parquet"
        if not path.exists():
            continue
        try:
            store = read_symbol_features(str(path), columns=names)
        except Exception:
            continue
        store.index = pd.to_datetime(store.index, utc=True, errors="coerce")
        source = store.reindex(frame.iloc[positions]["__ts__"])
        values[positions] = source.reindex(columns=names).to_numpy(
            dtype=np.float32, copy=False
        )
    additions = pd.DataFrame(values, columns=names, index=frame.index)
    out = pd.concat([frame, additions], axis=1, copy=False)
    return out, {
        name: float(np.isfinite(values[:, idx]).mean())
        for idx, name in enumerate(names)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--feature-root", type=Path, default=Path("data_perp/features/20260711_070000")
    )
    parser.add_argument("--end-exclusive", default="2026-07-11 10:00:00+00:00")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle: AlternativeMetaResidualBundle = joblib.load(args.bundle)
    existing = pd.read_parquet(args.existing)
    existing["__ts__"] = pd.to_datetime(existing["__ts__"], utc=True, errors="coerce")
    start = existing["__ts__"].max() + pd.Timedelta(hours=1)
    end = pd.Timestamp(args.end_exclusive)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    required = list(
        dict.fromkeys(bundle.required_input_features() + bundle.raw_selected_features)
    )
    live = _load_live_candidates(args.ledger, start=start, end=end, requested=required)
    if live.empty:
        raise ValueError(f"No live candidate rows in [{start}, {end})")
    live, store_coverage = _append_feature_store(
        live, feature_root=args.feature_root, requested=required
    )
    predictions = bundle.predict(live)
    lifecycle_sentinel = "oi_recovery_fraction_24h"
    market_sentinel = "mkt_median_oi_chg_4h_rz"
    live["feature_parity_eligible"] = (
        pd.to_numeric(live.get(lifecycle_sentinel), errors="coerce").notna()
        & pd.to_numeric(live.get(market_sentinel), errors="coerce").notna()
    )
    extension = live[
        [
            *KEYS,
            "decision_ts",
            "score_current_reference",
            "base_score",
            "production_policy_rank",
            "production_threshold_rank",
            "production_adjusted_rank",
            "feature_parity_eligible",
        ]
    ].copy()
    extension[predictions.columns] = predictions.to_numpy(dtype=np.float32, copy=False)
    extension["prediction_evidence"] = "frozen_live_candidate_handoff"
    extension["outcomes_available"] = False
    extension.to_parquet(
        args.output_dir / "july_prediction_extension.parquet",
        index=False,
        compression="zstd",
    )

    existing = existing.copy()
    existing["prediction_evidence"] = "research_handoff_with_outcomes"
    existing["outcomes_available"] = True
    combined = pd.concat([existing, extension], ignore_index=True, sort=False)
    combined = combined.sort_values(KEYS, kind="stable").drop_duplicates(
        KEYS, keep="last"
    )
    combined.to_parquet(
        args.output_dir / "july_predictions_combined.parquet",
        index=False,
        compression="zstd",
    )

    daily = (
        extension.assign(day=extension["__ts__"].dt.floor("D"))
        .groupby(["day", "side_name"], observed=True, sort=True)
        .agg(
            rows=("__ts__", "size"),
            timestamps=("__ts__", "nunique"),
            symbols=("__symbol__", "nunique"),
            parity_eligible_rate=("feature_parity_eligible", "mean"),
            current_score_mean=("score_current_reference", "mean"),
            alternative_score_mean=("score_shock_adjusted", "mean"),
            alternative_historical_rank_mean=("historical_rank", "mean"),
        )
        .reset_index()
    )
    daily.to_csv(args.output_dir / "prediction_counts_by_day_side.csv", index=False)

    generated_inputs = set(
        str(name)
        for name in (bundle.residual_representation_state or {}).get(
            "output_columns", []
        )
    )
    categorical_inputs = {"side_name", "archetype_policy_key"}
    coverage_inputs = [
        name for name in required if name not in generated_inputs | categorical_inputs
    ]
    feature_coverage = {
        name: float(pd.to_numeric(live[name], errors="coerce").notna().mean())
        if name in live.columns
        else 0.0
        for name in coverage_inputs
    }
    manifest = {
        "schema": "july_frozen_prediction_extension_v1",
        "existing_source": str(args.existing),
        "bundle_source": str(args.bundle),
        "bundle_fit_through": bundle.fit_through,
        "production_ledger_source": str(args.ledger),
        "feature_root": str(args.feature_root),
        "extension_start": live["__ts__"].min(),
        "extension_end": live["__ts__"].max(),
        "requested_end_exclusive": end,
        "extension_rows": int(len(extension)),
        "combined_rows": int(len(combined)),
        "extension_timestamps": int(extension["__ts__"].nunique()),
        "extension_days": int(extension["__ts__"].dt.floor("D").nunique()),
        "extension_symbols": int(extension["__symbol__"].nunique()),
        "feature_parity_eligible_rows": int(extension["feature_parity_eligible"].sum()),
        "feature_parity_eligible_rate": float(
            extension["feature_parity_eligible"].mean()
        ),
        "feature_parity_ineligible_reason": (
            "symbol excluded from the <=125 bps spread feature-generation universe"
        ),
        "required_feature_count": int(len(required)),
        "coverage_input_feature_count": int(len(coverage_inputs)),
        "required_feature_mean_coverage": float(
            np.mean(list(feature_coverage.values()))
        ),
        "required_features_below_90pct": sorted(
            name for name, rate in feature_coverage.items() if rate < 0.90
        ),
        "store_feature_coverage": store_coverage,
        "missing_timestamp_hours": [
            ts.isoformat()
            for ts in pd.date_range(
                existing["__ts__"].max().floor("h") + pd.Timedelta(hours=1),
                extension["__ts__"].max().floor("h"),
                freq="h",
                tz="UTC",
            ).difference(pd.DatetimeIndex(combined["__ts__"].dropna().unique()))
        ],
        "leakage_contract": (
            "Base/meta, residual recognizer, PCA, shock overlay, AE/GMM state, feature selection, "
            "HPO parameters, and historical rank reference are frozen through June. New July rows "
            "come from the production top-30 candidate handoff and contain no realized outcomes."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
