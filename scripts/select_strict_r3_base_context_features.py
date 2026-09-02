#!/usr/bin/env python3
"""Development-only stability selection for Strict-R3 context feature blocks.

This is deliberately a selector rather than a new model.  It scores a named
F2/F3/F4/F5 block using conditional mutual information (CMI) inside three
chronological 2025 development folds.  The condition is the already causal
prequential base-rank decile, so a selected field must carry information beyond
the inherited base coordinate.  No held 2025-Q4 or 2026 outcome is read.

The output is a deterministic, at-most-30-field contract.  It is only an input
to a subsequent strict-prequential base-only/base-and-residual rebuild; raw
context fields are never passed directly to MC1 from this selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features import (  # noqa: E402
    STRICT_R3_F2_ROLLING_CONTEXT_SOURCE_KEYS,
    STRICT_R3_F3_TRANSITION_SOURCE_KEYS,
    STRICT_R3_F4_EXECUTION_CONTEXT_SOURCE_KEYS,
    STRICT_R3_F5_ASSET_DIVERGENCE_SOURCE_KEYS,
    strict_r3_execution_divergence_features,
    strict_r3_rolling_context_features,
)
from scripts.run_strict_r3_base_recall_funnel import DEFAULT_SOURCE  # noqa: E402


DEVELOPMENT_FOLDS = (
    ("2025Q1", "2025-01-01", "2025-04-01"),
    ("2025Q2", "2025-04-01", "2025-07-01"),
    ("2025Q3", "2025-07-01", "2025-10-01"),
)
MAX_FIELDS = 30


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bin(values: pd.Series, bins: int = 10) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.notna()
    output = np.full(len(numeric), -1, dtype=np.int16)
    if int(finite.sum()) < bins * 3 or int(numeric[finite].nunique()) < 2:
        return output
    rank = numeric.loc[finite].rank(method="average", pct=True)
    output[finite.to_numpy()] = np.minimum(
        bins - 1, np.floor(rank.to_numpy(float) * bins).astype(np.int16),
    )
    return output


def _deterministic_cap(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    if len(frame) <= rows:
        return frame
    hashed = pd.util.hash_pandas_object(frame["candidate_id"], index=False).to_numpy(np.uint64)
    return frame.assign(__hash__=hashed).nsmallest(rows, "__hash__", keep="all").drop(columns="__hash__")


def _conditional_mi(feature: np.ndarray, base: np.ndarray, target: np.ndarray) -> float:
    total = 0.0
    weight = 0
    valid = (feature >= 0) & (base >= 0) & np.isfinite(target)
    for bucket in range(10):
        rows = valid & (base == bucket)
        count = int(rows.sum())
        if count < 30 or np.unique(target[rows]).size < 2:
            continue
        total += count * float(mutual_info_score(feature[rows], target[rows]))
        weight += count
    return total / weight if weight else float("nan")


def _conditional_direction(feature: np.ndarray, base: np.ndarray, target: np.ndarray) -> float:
    values: list[float] = []
    weights: list[int] = []
    valid = (feature >= 0) & (base >= 0) & np.isfinite(target)
    for bucket in range(10):
        rows = valid & (base == bucket)
        count = int(rows.sum())
        if count < 30:
            continue
        low = rows & (feature <= 1)
        high = rows & (feature >= 8)
        if not low.any() or not high.any():
            continue
        values.append(float(target[high].mean() - target[low].mean()))
        weights.append(count)
    return float(np.average(values, weights=weights)) if values else float("nan")


def _fields_and_features(source: pd.DataFrame, family: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if family in {"f2", "f3"}:
        derived = strict_r3_rolling_context_features(source)
        prefix = f"{family}_"
    elif family in {"f4", "f5"}:
        derived = strict_r3_execution_divergence_features(source)
        prefix = f"{family}_"
    else:
        raise ValueError(family)
    for name, value in derived.items():
        source[name] = value
    fields = tuple(name for name in source.columns if name.startswith(prefix))
    if not fields:
        raise ValueError(f"no generated {family} fields")
    return source, fields


def _source_columns(family: str) -> list[str]:
    raw = {
        "f2": STRICT_R3_F2_ROLLING_CONTEXT_SOURCE_KEYS,
        "f3": STRICT_R3_F3_TRANSITION_SOURCE_KEYS,
        "f4": STRICT_R3_F4_EXECUTION_CONTEXT_SOURCE_KEYS,
        "f5": STRICT_R3_F5_ASSET_DIVERGENCE_SOURCE_KEYS,
    }[family]
    return list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "r3_class", "r3_label_available_ts",
        "policy_net_bps", "policy_path_valid", "policy_label_available_ts",
        "prequential_base_rank42", *raw,
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("f2", "f3", "f4", "f5"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-rows-per-fold", type=int, default=250_000)
    parser.add_argument("--max-fields", type=int, default=MAX_FIELDS)
    args = parser.parse_args()
    if not 1 <= args.max_fields <= MAX_FIELDS:
        raise ValueError(f"max-fields must be in [1,{MAX_FIELDS}]")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    source = pd.read_parquet(args.source, columns=_source_columns(args.family))
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    source["r3_label_available_ts"] = pd.to_datetime(source["r3_label_available_ts"], utc=True, errors="coerce")
    source["policy_label_available_ts"] = pd.to_datetime(source["policy_label_available_ts"], utc=True, errors="coerce")
    source, fields = _fields_and_features(source, args.family)

    rows: list[dict[str, object]] = []
    for name, start, end in DEVELOPMENT_FOLDS:
        end_ts = _utc(end)
        fold = source.loc[
            source["__decision_ts__"].ge(_utc(start))
            & source["__decision_ts__"].lt(end_ts)
            & source["r3_label_available_ts"].lt(end_ts)
            & source["policy_label_available_ts"].lt(end_ts)
            & source["r3_class"].isin([0, 1, 2])
            & source["policy_path_valid"].fillna(False).astype(bool)
            & pd.to_numeric(source["policy_net_bps"], errors="coerce").notna()
            & pd.to_numeric(source["prequential_base_rank42"], errors="coerce").notna(),
            ["candidate_id", "r3_class", "policy_net_bps", "prequential_base_rank42", *fields],
        ].copy()
        fold = _deterministic_cap(fold, args.max_rows_per_fold)
        base = _bin(fold["prequential_base_rank42"])
        clear = fold["r3_class"].eq(2).to_numpy(np.int8)
        policy100 = pd.to_numeric(fold["policy_net_bps"], errors="coerce").ge(100.0).to_numpy(np.int8)
        for field in fields:
            feature = _bin(fold[field])
            rows.append({
                "family": args.family,
                "fold": name,
                "feature": field,
                "rows": int(len(fold)),
                "coverage": float((feature >= 0).mean()),
                "cmi_r3_clear": _conditional_mi(feature, base, clear),
                "cmi_policy_ge100": _conditional_mi(feature, base, policy100),
                "direction_policy_ge100": _conditional_direction(feature, base, policy100),
            })

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("feature", as_index=False, sort=True).agg(
        mean_coverage=("coverage", "mean"),
        cmi_r3_clear_mean=("cmi_r3_clear", "mean"),
        cmi_policy_ge100_mean=("cmi_policy_ge100", "mean"),
        cmi_positive_folds=("cmi_policy_ge100", lambda x: int(pd.Series(x).gt(0).sum())),
        direction_positive_folds=("direction_policy_ge100", lambda x: int(pd.Series(x).gt(0).sum())),
        direction_negative_folds=("direction_policy_ge100", lambda x: int(pd.Series(x).lt(0).sum())),
    )
    summary["selection_score"] = (
        summary["cmi_r3_clear_mean"].fillna(0.0)
        + summary["cmi_policy_ge100_mean"].fillna(0.0)
    )
    summary["stable"] = (
        summary["mean_coverage"].ge(.90)
        & summary["cmi_positive_folds"].ge(2)
        & summary["direction_positive_folds"].ge(2)
        & summary["direction_negative_folds"].le(1)
    )
    selected = summary.loc[summary["stable"]].sort_values(
        ["selection_score", "feature"], ascending=[False, True], kind="stable",
    ).head(args.max_fields).copy()
    if selected.empty:
        raise RuntimeError("no stable feature survived the predeclared development-only selector")

    args.out_dir.mkdir(parents=True)
    metrics.to_parquet(args.out_dir / "fold_conditional_mi.parquet", index=False)
    summary.to_parquet(args.out_dir / "feature_stability_summary.parquet", index=False)
    selected.to_parquet(args.out_dir / "selected_features.parquet", index=False)
    contract = {
        "schema": "strict_r3_context_feature_stability_selection_v1",
        "family": args.family,
        "selected_fields": selected["feature"].tolist(),
        "max_fields": int(args.max_fields),
        "selection_rule": "coverage>=90%; positive conditional policy>=100 CMI in >=2 development folds; positive conditional direction in >=2 folds; at most one negative direction fold; rank by mean CMI composite",
        "development_folds": [name for name, _, _ in DEVELOPMENT_FOLDS],
        "condition": "prequential_base_rank42 decile",
        "source_sha256": _sha256(args.source),
        "outcome_scope": "development-only; no 2025Q4 or 2026 labels were read",
        "downstream_rule": "selected raw fields may be tested in base and residual layers only; do not inject them directly into MC1 during the first funnel",
    }
    (args.out_dir / "feature_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "run_manifest.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "family": args.family, "selected": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
