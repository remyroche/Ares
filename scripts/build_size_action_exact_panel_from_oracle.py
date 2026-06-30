#!/usr/bin/env python3
"""Build a C3 size-action exact panel from action features and oracle labels.

The head-native C3el learner consumes a single ``size_action_exact_panel.csv``
that contains both deployable action features and exact-state counterfactual
labels. Some prospective/post-window runs materialize those pieces separately:

* ``action_feature_rows.parquet`` from the live-like action-feature builder;
* ``exact_state_counterfactual_labels.csv`` from the exact-state oracle.

This script joins them using the causal action key and writes the same exact
panel contract used by the training, support-audit, and promotion scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "is_baseline_action",
    "action_binds",
    "base_immediate_J",
    "action_immediate_J",
    "delta_immediate_J",
    "base_full_J",
    "action_full_J",
    "delta_full_J",
    "base_full_net_pnl",
    "action_full_net_pnl",
    "delta_full_net_pnl",
    "base_full_cost_pnl",
    "action_full_cost_pnl",
    "delta_full_cost_pnl",
    "base_full_turnover",
    "action_full_turnover",
    "delta_full_turnover",
    "base_immediate_trades",
    "action_immediate_trades",
]


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_key(frame: pd.DataFrame, *, multiplier_col: str) -> pd.DataFrame:
    required = {"timestamp", "strategy_id", multiplier_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame.shape=} missing required key columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["_multiplier_key"] = pd.to_numeric(out[multiplier_col], errors="coerce").round(8)
    out = out.loc[out["_multiplier_key"].notna()].copy()
    return out


def _to_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def build_exact_panel(
    *,
    action_features: Path,
    oracle_labels: Path,
    action_family: str = "size",
    feature_key_mode: str = "strict",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = _normalise_key(_read_frame(action_features), multiplier_col="multiplier")
    stale_label_cols = [col for col in LABEL_COLUMNS if col in features.columns]
    if stale_label_cols:
        features = features.drop(columns=stale_label_cols)
    labels_raw = _read_frame(oracle_labels)
    if "action_family" not in labels_raw.columns:
        raise ValueError("oracle labels are missing action_family")
    labels = labels_raw.loc[labels_raw["action_family"].astype(str).eq(str(action_family))].copy()
    labels = _normalise_key(labels, multiplier_col="action_value")

    feature_dupes = int(features.duplicated(["timestamp", "strategy_id", "_multiplier_key"]).sum())
    label_dupes = int(labels.duplicated(["timestamp", "strategy_id", "_multiplier_key"]).sum())
    if feature_dupes:
        raise ValueError(f"action features contain duplicate action keys: {feature_dupes}")
    if label_dupes:
        raise ValueError(f"oracle labels contain duplicate size-action keys: {label_dupes}")
    if feature_key_mode not in {"strict", "labels"}:
        raise ValueError(f"feature_key_mode must be 'strict' or 'labels', got {feature_key_mode!r}")
    feature_rows_before_filter = int(len(features))
    if feature_key_mode == "labels":
        label_keys = labels[["timestamp", "strategy_id", "_multiplier_key"]].drop_duplicates()
        features = features.merge(label_keys, on=["timestamp", "strategy_id", "_multiplier_key"], how="inner")

    label_cols = ["timestamp", "strategy_id", "_multiplier_key", *[c for c in LABEL_COLUMNS if c in labels.columns]]
    merged = features.merge(
        labels[label_cols],
        on=["timestamp", "strategy_id", "_multiplier_key"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        sample = merged.loc[merged["_merge"].ne("both"), ["timestamp", "strategy_id", "multiplier"]].head(10)
        raise ValueError(
            "action features were not fully matched to exact-state size labels: "
            f"{unmatched} unmatched rows. Sample: {sample.to_dict(orient='records')}"
        )

    merged = merged.drop(columns=["_merge", "_multiplier_key"])
    for col in LABEL_COLUMNS:
        if col not in merged.columns:
            continue
        if col in {"is_baseline_action", "action_binds"}:
            merged[col] = _to_bool_series(merged[col])
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    affected = pd.to_numeric(merged.get("affected_notional"), errors="coerce").fillna(0.0)
    denom = np.maximum(affected.to_numpy(dtype=float), 1.0)
    merged["delta_full_J_per_notional"] = pd.to_numeric(merged["delta_full_J"], errors="coerce").fillna(0.0).to_numpy(
        dtype=float
    ) / denom
    merged["delta_immediate_J_per_notional"] = pd.to_numeric(
        merged["delta_immediate_J"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float) / denom

    audit = {
        "action_features": str(action_features),
        "oracle_labels": str(oracle_labels),
        "action_family": str(action_family),
        "feature_rows": int(len(features)),
        "feature_rows_before_filter": feature_rows_before_filter,
        "feature_key_mode": str(feature_key_mode),
        "label_rows_raw": int(len(labels_raw)),
        "label_rows_family": int(len(labels)),
        "stale_feature_label_columns_dropped": stale_label_cols,
        "output_rows": int(len(merged)),
        "unique_groups": int(merged[["timestamp", "strategy_id"]].drop_duplicates().shape[0]),
        "timestamp_min": str(merged["timestamp"].min()) if not merged.empty else "",
        "timestamp_max": str(merged["timestamp"].max()) if not merged.empty else "",
        "multipliers": sorted(float(x) for x in pd.to_numeric(merged["multiplier"], errors="coerce").dropna().unique()),
        "positive_delta_full_rows": int(pd.to_numeric(merged["delta_full_J"], errors="coerce").fillna(0.0).gt(0.0).sum()),
        "binding_rows": int(merged["action_binds"].astype(bool).sum()) if "action_binds" in merged.columns else 0,
    }
    return merged.sort_values(["timestamp", "strategy_id", "multiplier"]).reset_index(drop=True), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--oracle-labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-family", default="size")
    parser.add_argument(
        "--feature-key-mode",
        choices=["strict", "labels"],
        default="strict",
        help="Use 'labels' to filter action features down to oracle-labeled keys before joining.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel, audit = build_exact_panel(
        action_features=args.action_features,
        oracle_labels=args.oracle_labels,
        action_family=args.action_family,
        feature_key_mode=args.feature_key_mode,
    )
    panel.to_csv(args.out_dir / "size_action_exact_panel.csv", index=False)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "build_size_action_exact_panel_from_oracle",
                **audit,
                "outputs": {"size_action_exact_panel": str(args.out_dir / "size_action_exact_panel.csv")},
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    pd.DataFrame([audit]).to_csv(args.out_dir / "merge_audit.csv", index=False)
    print(json.dumps(audit, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
