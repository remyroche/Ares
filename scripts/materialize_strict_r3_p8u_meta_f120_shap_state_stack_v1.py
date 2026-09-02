#!/usr/bin/env python3
"""Materialise the additive F120 + explanation/state Meta feature stack.

This is a target-free feature producer.  It deliberately retains every field
from the frozen 120-field Under parent and appends, by exact identity:

* F72 TreeSHAP attribution summaries and signed contributions;
* prior-resolved same-rule SHAP support/error state;
* the preselected causal Kalman / fast--slow / transition and synergy state;
* global prior-only innovation Mahalanobis and innovation-dispersion fields.

``inclusion_uplift`` is intentionally *not* a candidate input: it is a
training-only feature-selection statistic, calculated fold-locally later in
the Meta funnel.  Persisting it per row would be a semantic error.

The producer never opens policy, path, outcome, MC1, portfolio, or exchange
inputs.  It produces an append-only research overlay only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_f120_shap_state_stack_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
FORBIDDEN = (
    "policy_", "supportive_", "h12_", "label_", "outcome_", "mfe", "mae",
)
FAST_SLOW_PAIRS = ((2, 14), (3, 14), (3, 21), (5, 21), (5, 42), (7, 42))


def _once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value}-01", tz="UTC")


def _months(values: str) -> tuple[pd.Timestamp, ...]:
    start, finish = (_month(token.strip()) for token in values.split(":", 1))
    result: list[pd.Timestamp] = []
    current = start
    while current < finish:
        result.append(current)
        current += pd.offsets.MonthBegin(1)
    if not result:
        raise ValueError("--months must contain at least one month")
    return tuple(result)


def _names(path: Path) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _target_free_schema(path: Path, fields: Iterable[str]) -> None:
    bad = [field for field in fields if field.lower().startswith(FORBIDDEN)]
    if bad:
        raise AssertionError(f"{path}: prohibited target/outcome fields {bad[:8]}")


def _read(path: Path, fields: list[str]) -> pd.DataFrame:
    values = pd.read_parquet(path, columns=[*IDENTITY, *fields])
    values["__decision_ts__"] = pd.to_datetime(values["__decision_ts__"], utc=True, errors="raise")
    if values.duplicated(list(IDENTITY)).any() or not values.side_name.eq("long").all():
        raise AssertionError(f"{path}: nonunique or non-long target-free identity")
    return values.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _read_state(path: Path, fields: list[str]) -> pd.DataFrame:
    values = pd.read_parquet(path, columns=["__decision_ts__", *fields])
    values["__decision_ts__"] = pd.to_datetime(values["__decision_ts__"], utc=True, errors="raise")
    if values.__decision_ts__.duplicated().any():
        raise AssertionError(f"{path}: duplicate hourly state timestamp")
    return values.sort_values("__decision_ts__", kind="stable").reset_index(drop=True)


def _exact_join(left: pd.DataFrame, right: pd.DataFrame, *, on: list[str], label: str) -> pd.DataFrame:
    merged = left.merge(right, on=on, how="left", validate="one_to_one" if on == list(IDENTITY) else "many_to_one")
    if len(merged) != len(left):
        raise AssertionError(f"{label}: join changed candidate identity count")
    added = [field for field in right.columns if field not in on]
    if added and merged.loc[:, added].isna().all(axis=None):
        raise AssertionError(f"{label}: no joined target-free values")
    return merged


def _coverage(frame: pd.DataFrame, fields: Iterable[str], month: pd.Timestamp) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce")
        rows.append({
            "month": f"{month:%Y-%m}", "feature": field, "rows": int(len(frame)),
            "finite_fraction": float(np.isfinite(values).mean()),
            "nunique": int(values.nunique(dropna=True)), "variance": float(values.var(skipna=True)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--shap-root", type=Path, required=True)
    parser.add_argument("--shap-state-root", type=Path, required=True)
    parser.add_argument("--state-overlay-root", type=Path, required=True)
    parser.add_argument("--market-state-root", type=Path, required=True)
    parser.add_argument("--months", default="2025-08:2026-08", help="inclusive:exclusive YYYY-MM range")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    months = _months(args.months)
    parent_contract = json.loads(args.parent_contract.read_text())
    parent_fields = list(map(str, parent_contract.get("selected_features", ())))
    if len(parent_fields) != 120 or len(set(parent_fields)) != 120:
        raise AssertionError("expected the exact frozen 120-field parent contract")
    _target_free_schema(args.parent_contract, parent_fields)

    sample_month = months[-1]
    shap_sample = args.shap_root / f"month={sample_month:%Y-%m}.parquet"
    shap_state_sample = args.shap_state_root / f"month={sample_month:%Y-%m}" / "causal_feature_universe.parquet"
    overlay_sample = args.state_overlay_root / "features" / f"month={sample_month:%Y-%m}" / "causal_feature_universe.parquet"
    market_sample = args.market_state_root / "market_state_hourly.parquet"
    for path in (shap_sample, shap_state_sample, overlay_sample, market_sample):
        if not path.exists():
            raise FileNotFoundError(path)
    shap_fields = [field for field in _names(shap_sample) if field.startswith("shap_f72_") and not field.endswith("_parent_score")]
    shap_state_fields = [field for field in _names(shap_state_sample) if field.startswith("shap_top")]
    state_overlay_fields = [field for field in _names(overlay_sample) if field.startswith("ms_")]
    innovation_fields = [
        f"ms_global_fast{fast}d_slow{slow}d_innovation_{suffix}"
        for fast, slow in FAST_SLOW_PAIRS for suffix in ("mahalanobis", "dispersion")
    ]
    state_names = set(_names(market_sample))
    missing_innovation = sorted(set(innovation_fields).difference(state_names))
    if missing_innovation:
        raise AssertionError(f"market state lacks requested innovation fields {missing_innovation}")
    selected = [*parent_fields, *shap_fields, *shap_state_fields, *state_overlay_fields, *innovation_fields]
    if len(selected) != len(set(selected)):
        duplicate = sorted({field for field in selected if selected.count(field) > 1})
        raise AssertionError(f"feature stack duplicates fields: {duplicate}")
    _target_free_schema(shap_sample, shap_fields)
    _target_free_schema(shap_state_sample, shap_state_fields)
    _target_free_schema(overlay_sample, state_overlay_fields)
    _target_free_schema(market_sample, innovation_fields)

    out.mkdir(parents=True)
    contracts = {
        "schema": SCHEMA,
        "parent_f120": parent_fields,
        "shap_attribution": shap_fields,
        "shap_same_rule_state": shap_state_fields,
        "kalman_fast_slow_transition_and_synergy": state_overlay_fields,
        "market_innovation_mahalanobis_and_dispersion": innovation_fields,
        "all_features": selected,
        "inclusion_uplift": "training-only fold-local selector statistic; deliberately not an inference feature",
        "kalman_contract": "predeclared half-life pairs from the target-free market-state materializer; tuning is a feature-family/block-selection question, not an in-sample per-row transform",
    }
    _once(out / "feature_contract.json", contracts)
    coverage: list[dict[str, object]] = []
    source_audit: list[dict[str, object]] = []
    state = _read_state(market_sample, innovation_fields)
    for month in months:
        parent_path = args.parent_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        shap_path = args.shap_root / f"month={month:%Y-%m}.parquet"
        shap_state_path = args.shap_state_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        overlay_path = args.state_overlay_root / "features" / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        for path in (parent_path, shap_path, shap_state_path, overlay_path):
            if not path.exists():
                raise FileNotFoundError(path)
        parent = _read(parent_path, parent_fields)
        combined = _exact_join(parent, _read(shap_path, shap_fields), on=list(IDENTITY), label=f"shap {month:%Y-%m}")
        combined = _exact_join(combined, _read(shap_state_path, shap_state_fields), on=list(IDENTITY), label=f"shap-state {month:%Y-%m}")
        combined = _exact_join(combined, _read(overlay_path, state_overlay_fields), on=list(IDENTITY), label=f"kalman-state {month:%Y-%m}")
        local_state = state.loc[state.__decision_ts__.ge(month) & state.__decision_ts__.lt(month + pd.offsets.MonthBegin(1))]
        combined = _exact_join(combined, local_state, on=["__decision_ts__"], label=f"innovation-state {month:%Y-%m}")
        if list(combined.columns) != [*IDENTITY, *selected]:
            raise AssertionError(f"{month:%Y-%m}: feature ordering drift")
        destination = out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(destination, index=False, compression="zstd")
        coverage.extend(_coverage(combined, selected, month))
        source_audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(combined)), "parent": str(parent_path),
            "shap": str(shap_path), "shap_state": str(shap_state_path), "state_overlay": str(overlay_path),
            "candidate_identity_exact": True, "hourly_state_identity_exact": True,
        })
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_parquet(out / "feature_coverage.parquet", index=False, compression="zstd")
    pd.DataFrame(source_audit).to_parquet(out / "source_identity_audit.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "parent_f120_contract_is_retained_unchanged": True,
        "all_additions_are_target_free": True,
        "candidate_additions_join_by_exact_identity": True,
        "market_state_additions_join_by_exact_timestamp": True,
        "no_policy_path_outcome_mc1_portfolio_or_exchange_source_opened": True,
        "kalman_pairs_are_predeclared_and_not_refit_per_fold": True,
        "inclusion_uplift_is_training_only_not_an_inference_feature": True,
        "feature_order_is_frozen": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline target-free append-only Meta feature overlay",
        "months": [f"{month:%Y-%m}" for month in months], "feature_count": len(selected),
        "feature_contract_sha256": _sha(out / "feature_contract.json"),
        "sources": {
            "parent_root": str(args.parent_root.resolve()), "parent_contract": str(args.parent_contract.resolve()),
            "shap_root": str(args.shap_root.resolve()), "shap_state_root": str(args.shap_state_root.resolve()),
            "state_overlay_root": str(args.state_overlay_root.resolve()), "market_state_root": str(args.market_state_root.resolve()),
        }, "source_hashes": {"parent_contract": _sha(args.parent_contract.resolve())},
        "families": {key: len(value) if isinstance(value, list) else value for key, value in contracts.items() if key != "all_features"},
    })
    print(json.dumps({"out": str(out), "months": len(months), "feature_count": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
