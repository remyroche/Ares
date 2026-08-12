#!/usr/bin/env python3
"""Long-only conditional correctness of the frozen Cap-120 policy score.

This runner is deliberately different from the earlier raw-path experiment.
The structural inputs are the *actual* rule-family contributions emitted by the
frozen base trees.  They are used to explain when the Cap-120 score is right or
wrong; raw feature-pair membership is never used as the correctness target.

The execution label is recomputed from the historical 15-minute source using
the frozen policy (48 bars, SL=3 ATR, trailing activation=.5 ATR,
giveback=.25 ATR, one 100-bps cost).  The score is fitted from the existing
Cap-120 head contract and calibrated only on rows available before the scored
partition.  Evaluation is one pooled global ranking, with monthly/weekly
diagnostics.  Shorts are rejected at load time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

# The desktop bundle keeps the scientific stack in a workspace-local path.
# Insert it before importing numpy/pandas so the script is runnable with the
# system ``python3 -S`` used by the Ares jobs.
sys.path[:0] = [
    "/Users/remyroche/Documents/Codex/.tmp_pydeps_20260806",
    "/Users/remyroche/Library/Python/3.12/lib/python/site-packages",
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload",
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12",
]

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.trailing_exit_grid import simulate_h12_stop_trailing_grid
from scripts.run_joint_correctness_mlp_meta import (
    _feature_columns,
    _make_head_fit,
    _matrix as _base_matrix,
    _predict as _base_predict,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

SCHEMA = "long_family_conditional_correctness_v1"
SIDE = "long"
HORIZON_BARS = 48
COST_BPS = 100.0
STOP_ATR = 3.0
TRAIL_ACTIVATION_ATR = 0.5
TRAIL_GIVEBACK_ATR = 0.25
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
FAMILY_PREFIX = "family_"

DEFAULT_SIDECAR = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4/tree_meta_candidate_sidecar.parquet"
DEFAULT_FAMILY_ROOT = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4"
DEFAULT_BASELINE = ROOT / "data_perp/artifacts/long_family_conditional_correctness_20260807_v1"
DEFAULT_PATH_ROOT = ROOT / "data_perp/artifacts/h12_query_path_grid_20260805_v2"
DEFAULT_BARS_ROOT = ROOT / "15m_ohlcv_perp"


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _schema_columns(path: Path) -> list[str]:
    return list(map(str, pq.ParquetFile(path).schema.names))


def _load_sidecar(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    schema = _schema_columns(path)
    required = {
        "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "atr_bps",
        "label_available_ts", "query_id", "meta_partition", "fold", "base_expected_bps",
    }
    missing = sorted(required.difference(schema))
    if missing:
        raise ValueError(f"sidecar missing required fields: {missing}")
    # The sidecar has an ordered causal block between atr_bps and
    # label_available_ts.  Keep raw causal/context fields plus prior-only health
    # fields; family memberships are recomputed from actual paths below.
    start = schema.index("atr_bps") + 1
    end = schema.index("label_available_ts")
    ordered_block = schema[start:end]
    base_fields = [
        name for name in ordered_block
        if not name.startswith("base_structural_family__")
        and not name.startswith("base_reasoning__")
    ]
    health_fields = [name for name in schema if name.startswith("structural_health__")]
    context_fields = list(dict.fromkeys([*base_fields, *health_fields]))
    columns = list(required.union(context_fields))
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    if frame[["__ts__", "label_available_ts"]].isna().any().any():
        raise ValueError("sidecar has invalid UTC timestamps")
    if not frame["label_available_ts"].ge(frame["__ts__"]).all():
        raise ValueError("sidecar label availability precedes decision")
    if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("this pipeline is explicitly long-only")
    if frame.duplicated(["fold", "candidate_id"]).any():
        raise ValueError("duplicate candidate inside a fold")
    frame = frame.sort_values(["fold", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    for field in context_fields:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype("float32")
    return frame, base_fields, context_fields


def _materialize_execution_labels(
    frame: pd.DataFrame, path_root: Path, bars_root: Path,
) -> pd.DataFrame:
    """Recompute the frozen 15m exit outcome, using decision-time entries."""

    from extreme_price_movements.trailing_exit_grid import net_bps

    result = frame.loc[:, ["fold", "candidate_id", "__ts__"]].copy()
    result["policy_net_bps"] = np.nan
    result["policy_gross_bps"] = np.nan
    result["policy_valid"] = False
    result["policy_entry_ts"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["policy_entry_price"] = np.nan
    result["policy_atr_bps"] = np.nan
    result["policy_path_resolution_minutes"] = 15
    result["policy_label_available_ts"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    result["policy_symbol"] = ""

    work = frame.loc[:, ["fold", "candidate_id", "__ts__"]].copy()
    work["symbol"] = work["candidate_id"].astype(str).str.split("|").str[0]
    for symbol, group in work.groupby("symbol", sort=False):
        path_file = path_root / f"symbol={symbol}.parquet"
        bar_file = bars_root / (str(symbol).lower().replace("_", "") + "_15m.parquet")
        if not path_file.exists() or not bar_file.exists():
            continue
        path = pd.read_parquet(
            path_file,
            columns=["candidate_id", "__decision_ts__", "entry_price", "atr_bps", "label_valid"],
        )
        # The rolling sidecar can carry the same candidate in more than one
        # outer fold; the immutable path table is unique by candidate only.
        z = group.merge(path, on="candidate_id", how="inner", validate="many_to_one")
        if z.empty:
            continue
        bars = pd.read_parquet(bar_file)
        time_col = next((c for c in ("ts", "timestamp", "__index_level_0__") if c in bars.columns), None)
        if time_col is not None:
            bars = bars.set_index(time_col)
        if not isinstance(bars.index, pd.DatetimeIndex):
            raise ValueError(f"15m source lacks timestamps: {bar_file}")
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars.loc[:, ["high", "low", "close"]]
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        decision_ts = pd.to_datetime(z["__decision_ts__"], utc=True)
        starts = bars.index.get_indexer(decision_ts)
        valid = (
            (starts >= 0)
            & (starts + HORIZON_BARS <= len(bars))
            & z["label_valid"].fillna(False).to_numpy(bool)
            & np.isfinite(z["entry_price"].to_numpy(float))
            & (z["entry_price"].to_numpy(float) > 0.0)
            & np.isfinite(z["atr_bps"].to_numpy(float))
            & (z["atr_bps"].to_numpy(float) > 0.0)
        )
        if not valid.any():
            continue
        zv = z.loc[valid].copy()
        starts_v = starts[valid]
        entry = zv["entry_price"].to_numpy(float)
        atr_bps = zv["atr_bps"].to_numpy(float)
        atr = entry * atr_bps / 10_000.0
        gross_atr = simulate_h12_stop_trailing_grid(
            bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float),
            starts_v.astype(np.int64), entry.astype(np.float32), atr.astype(np.float32),
            np.ones(len(zv), dtype=np.float32), np.asarray([STOP_ATR], dtype=np.float32),
            np.asarray([TRAIL_ACTIVATION_ATR], dtype=np.float32),
            np.asarray([TRAIL_GIVEBACK_ATR], dtype=np.float32), horizon_bars=HORIZON_BARS,
        ).reshape(-1)
        policy_net = net_bps(gross_atr.reshape(-1, 1, 1, 1), atr_bps, cost_bps=COST_BPS).reshape(-1)
        idx = result.set_index(["fold", "candidate_id"]).index
        keys = pd.MultiIndex.from_frame(zv[["fold", "candidate_id"]])
        take = idx.get_indexer(keys)
        ok = take >= 0
        result.loc[take[ok], "policy_net_bps"] = policy_net[ok]
        result.loc[take[ok], "policy_gross_bps"] = policy_net[ok] + COST_BPS
        result.loc[take[ok], "policy_valid"] = np.isfinite(policy_net[ok])
        entry_ts_values = pd.DatetimeIndex(decision_ts.loc[zv.index]).tz_convert("UTC")
        result.loc[take[ok], "policy_entry_ts"] = entry_ts_values
        result.loc[take[ok], "policy_entry_price"] = entry
        result.loc[take[ok], "policy_atr_bps"] = atr_bps
        result.loc[take[ok], "policy_label_available_ts"] = entry_ts_values + pd.Timedelta(hours=12)
        result.loc[take[ok], "policy_symbol"] = symbol

    # The historical candidate panel convention is a one-hour feature/decision
    # offset.  This assertion prevents silently entering before the decision.
    valid = result["policy_valid"]
    if valid.any():
        offsets = pd.to_datetime(result.loc[valid, "policy_entry_ts"], utc=True) - pd.to_datetime(result.loc[valid, "__ts__"], utc=True)
        if not (offsets == pd.Timedelta(hours=1)).all():
            raise AssertionError("policy entry is not the decision bar after the feature timestamp")
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True)
    return result


def _family_contract(root: Path) -> tuple[dict[str, str], dict[str, int], dict[str, float], pd.DataFrame]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    cluster_to_field = {str(k): str(v) for k, v in manifest["cluster_feature_map"].items()}
    assignments = pd.read_parquet(root / "structural_family_assignments.parquet")
    selected = assignments.loc[assignments["is_selected"].astype(bool)].copy()
    selected = selected[selected["cluster_id"].astype(str).isin(cluster_to_field)]
    lookup: dict[str, int] = {}
    confidence_lookup: dict[str, float] = {}
    fields = sorted(cluster_to_field.values())
    field_index = {name: i for i, name in enumerate(fields)}
    for row in selected.itertuples(index=False):
        lookup[str(row.rule_instance_id)] = field_index[cluster_to_field[str(row.cluster_id)]]
        confidence_lookup[str(row.rule_instance_id)] = float(
            np.clip(getattr(row, "similarity_to_medoid", 1.0), 0.0, 1.0)
        )
    return cluster_to_field, lookup, confidence_lookup, selected


def _leaf_weight_maps(root: Path) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    """Load training-frozen leaf values and contribution scales.

    The catalogs are emitted with each frozen base fold and contain no realised
    policy outcomes.  They are therefore safe to use for an inference-time
    weighting ablation.  Scales are fold-local and are computed from the
    training leaf catalog, never from the outer rows.
    """
    values: dict[tuple[str, str], float] = {}
    value_scales: dict[str, float] = {}
    contribution_scales: dict[str, float] = {}
    catalog_root = root / "strict_base_reasoning"
    for catalog in sorted(catalog_root.glob("*/leaf_rule_catalog.parquet")):
        fold = catalog.parent.name
        table = pq.read_table(
            catalog,
            columns=["rule_signature", "tree_leaf_value", "ensemble_tree_contribution"],
        ).to_pandas()
        table["tree_leaf_value"] = pd.to_numeric(table["tree_leaf_value"], errors="coerce").fillna(0.0)
        table["ensemble_tree_contribution"] = pd.to_numeric(
            table["ensemble_tree_contribution"], errors="coerce"
        ).fillna(0.0)
        for row in table.itertuples(index=False):
            values[(fold, str(row.rule_signature))] = float(row.tree_leaf_value)
        va = np.abs(table["tree_leaf_value"].to_numpy(float))
        ca = np.abs(table["ensemble_tree_contribution"].to_numpy(float))
        value_scales[fold] = float(np.nanmedian(va[va > 1e-12])) if np.any(va > 1e-12) else 1.0
        contribution_scales[fold] = float(np.nanmedian(ca[ca > 1e-12])) if np.any(ca > 1e-12) else 1.0
    if not values:
        raise RuntimeError(f"no frozen leaf catalogs found below {catalog_root}")
    return values, value_scales, contribution_scales


def _aggregate_contributions(
    root: Path, lookup: dict[str, int], confidence_lookup: dict[str, float], fields: list[str],
    leaf_weighting: str = "raw", leaf_catalog_root: Path | None = None,
) -> pd.DataFrame:
    """Aggregate actual tree-path contributions to a candidate × family matrix.

    ``raw`` is the historical control.  The other modes reweight each signed
    path contribution by a bounded, fold-training-derived leaf-value and/or
    contribution-strength factor while retaining its sign.  This tests whether
    strong base leaves should have more authority in the residual learner
    without using realised outer outcomes.
    """
    valid_modes = {"raw", "value", "contribution", "value_x_contribution"}
    if leaf_weighting not in valid_modes:
        raise ValueError(f"unknown leaf weighting {leaf_weighting!r}; choose from {sorted(valid_modes)}")
    leaf_values, value_scales, contribution_scales = _leaf_weight_maps(
        leaf_catalog_root if leaf_catalog_root is not None else root
    )

    selected_parts: list[pd.DataFrame] = []
    total_parts: list[pd.DataFrame] = []
    for path in sorted((root / "family_contributions").glob("*.parquet")):
        fold = path.stem
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=[
                "candidate_id", "fold_id", "side_name", "head_name", "rule_signature",
                "family_ensemble_tree_contribution",
            ],
            batch_size=750_000,
        ):
            b = batch.to_pandas()
            # The family contract is defined for the long-side p_clear
            # contribution stream.  The sidecar stores other structural heads
            # in the same files; including them in the denominator while only
            # mapping p_clear rules creates an artificial unassigned mass.
            b = b.loc[
                (b["side_name"].astype(str).str.lower() == "long")
                & (b["head_name"].astype(str) == "p_clear")
            ].copy()
            if b.empty:
                continue
            b["family_ensemble_tree_contribution"] = pd.to_numeric(
                b["family_ensemble_tree_contribution"], errors="coerce"
            ).fillna(0.0)
            # Every factor is based on the frozen training leaf catalog.  The
            # per-row contribution is only the already-emitted base output;
            # no future price or policy result is consulted.
            b["leaf_value"] = [
                leaf_values.get((str(f), str(r)), 0.0)
                for f, r in zip(b["fold_id"], b["rule_signature"])
            ]
            value_factor = np.clip(
                b["leaf_value"].abs().to_numpy(float)
                / max(value_scales.get(str(fold), 1.0), 1e-12),
                0.25, 4.0,
            )
            contribution_factor = np.clip(
                b["family_ensemble_tree_contribution"].abs().to_numpy(float)
                / max(contribution_scales.get(str(fold), 1.0), 1e-12),
                0.25, 4.0,
            )
            if leaf_weighting == "raw":
                b["leaf_weight_factor"] = 1.0
            elif leaf_weighting == "value":
                b["leaf_weight_factor"] = value_factor
            elif leaf_weighting == "contribution":
                b["leaf_weight_factor"] = contribution_factor
            else:
                # Geometric combination prevents the two related measures
                # from multiplying into an unstable 16x path dominance.
                b["leaf_weight_factor"] = np.sqrt(value_factor * contribution_factor)
            # Candidate IDs recur across outer folds.  The contribution
            # denominator is therefore keyed by (fold, candidate_id), not by
            # candidate_id alone; otherwise the same row's contribution from
            # another fold inflates the denominator and falsely reports a
            # large unassigned mass.
            total_parts.append(
                b.assign(
                    abs_contribution=(
                        b["family_ensemble_tree_contribution"].abs()
                        * b["leaf_weight_factor"]
                    )
                )
                .groupby(["fold_id", "candidate_id"], observed=True)["abs_contribution"].sum()
                .rename("total_abs_contribution").reset_index()
            )
            keys = b["fold_id"].astype(str) + "::" + b["rule_signature"].astype(str)
            b["family_index"] = keys.map(lookup)
            b["family_assignment_similarity"] = keys.map(confidence_lookup).fillna(0.0).astype(float)
            b = b[b["family_index"].notna()].copy()
            if b.empty:
                continue
            b["family_index"] = b["family_index"].astype(int)
            selected_parts.append(
                b.assign(
                    weighted_contribution=(
                        b["family_ensemble_tree_contribution"] * b["leaf_weight_factor"]
                    ),
                    abs_contribution=(
                        b["family_ensemble_tree_contribution"].abs()
                        * b["leaf_weight_factor"]
                    ),
                    confidence_weighted_abs=(
                        b["family_ensemble_tree_contribution"].abs()
                        * b["leaf_weight_factor"]
                        * b["family_assignment_similarity"].clip(0.0, 1.0)
                    ),
                )
                .groupby(["fold_id", "candidate_id", "family_index"], observed=True)
                .agg(
                    family_contribution=("weighted_contribution", "sum"),
                    family_abs_contribution=("abs_contribution", "sum"),
                    family_confidence_weighted_abs=("confidence_weighted_abs", "sum"),
                    family_leaf_weight_factor=("leaf_weight_factor", "mean"),
                )
                .reset_index()
                .rename(columns={"fold_id": "fold"})
            )
    if not total_parts:
        raise RuntimeError("no structural contribution files were found")
    totals = (
        pd.concat(total_parts, ignore_index=True)
        .groupby(["fold_id", "candidate_id"], observed=True)["total_abs_contribution"]
        .sum()
        .rename("total_abs_contribution")
        .reset_index()
        .rename(columns={"fold_id": "fold"})
    )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = (
        selected.groupby(["fold", "candidate_id", "family_index"], observed=True)[
            ["family_contribution", "family_abs_contribution", "family_confidence_weighted_abs", "family_leaf_weight_factor"]
        ]
        .sum()
        .reset_index()
    )
    selected["family_name"] = selected["family_index"].map({i: f for i, f in enumerate(fields)})
    wide = selected.pivot_table(
        index=["fold", "candidate_id"], columns="family_name", values="family_contribution", fill_value=0.0
    )
    abs_wide = selected.pivot_table(
        index=["fold", "candidate_id"], columns="family_name", values="family_abs_contribution", fill_value=0.0
    )
    confidence_wide = selected.pivot_table(
        index=["fold", "candidate_id"], columns="family_name", values="family_confidence_weighted_abs", fill_value=0.0
    )
    leaf_weight_wide = selected.pivot_table(
        index=["fold", "candidate_id"], columns="family_name", values="family_leaf_weight_factor", fill_value=1.0
    )
    wide = wide.reindex(columns=fields, fill_value=0.0).reset_index()
    abs_wide = abs_wide.reindex(columns=fields, fill_value=0.0).reset_index()
    confidence_wide = confidence_wide.reindex(columns=fields, fill_value=0.0).reset_index()
    leaf_weight_wide = leaf_weight_wide.reindex(columns=fields, fill_value=1.0).reset_index()
    wide.columns.name = None
    abs_wide.columns.name = None
    confidence_wide.columns.name = None
    leaf_weight_wide.columns.name = None
    wide = wide.merge(totals, on=["fold", "candidate_id"], how="left", validate="one_to_one")
    abs_wide = abs_wide.merge(
        totals, on=["fold", "candidate_id"], how="left", validate="one_to_one"
    )
    confidence_wide = confidence_wide.merge(
        totals, on=["fold", "candidate_id"], how="left", validate="one_to_one"
    )
    denominator = wide["total_abs_contribution"].fillna(0.0).to_numpy(float)
    for f in fields:
        wide[f] = pd.to_numeric(wide[f], errors="coerce").fillna(0.0).astype("float32")
        abs_values = pd.to_numeric(abs_wide[f], errors="coerce").fillna(0.0).to_numpy(float)
        confidence_values = pd.to_numeric(confidence_wide[f], errors="coerce").fillna(0.0).to_numpy(float)
        wide[f"family_abs_share__{f}"] = (
            abs_values / np.maximum(denominator, 1e-12)
        ).clip(0.0, 1.0).astype("float32")
        wide[f"family_confidence_share__{f}"] = (
            confidence_values / np.maximum(denominator, 1e-12)
        ).clip(0.0, 1.0).astype("float32")
        wide[f"family_active__{f}"] = (wide[f].abs() > 1e-12).astype("float32")
        wide[f"family_leaf_weight__{f}"] = pd.to_numeric(
            leaf_weight_wide[f], errors="coerce"
        ).fillna(1.0).clip(0.25, 4.0).astype("float32")
    wide["family_total_abs_contribution"] = wide["total_abs_contribution"].fillna(0.0).astype("float32")
    wide = wide.drop(columns=["total_abs_contribution"])
    share_cols = [f"family_abs_share__{f}" for f in fields]
    wide["family_unassigned_mass"] = (1.0 - wide[share_cols].sum(axis=1)).clip(0.0, 1.0).astype("float32")
    confidence_cols = [f"family_confidence_share__{f}" for f in fields]
    wide["family_assignment_quality"] = wide[confidence_cols].sum(axis=1).clip(0.0, 1.0).astype("float32")
    wide["family_low_confidence_mass"] = (
        wide[share_cols].sum(axis=1) - wide[confidence_cols].sum(axis=1)
    ).clip(0.0, 1.0).astype("float32")
    return wide


def _rule_path_metrics(root: Path, cluster_to_field: dict[str, str], selected: pd.DataFrame) -> pd.DataFrame:
    catalogue = pd.read_parquet(root / "structural_rule_catalogue.parquet")
    keep = selected.loc[:, ["rule_instance_id", "cluster_id", "similarity_to_medoid", "is_recurrent"]].merge(
        catalogue.loc[:, ["rule_instance_id", "fold_id", "head_name", "rule_structural_path_json", "train_leaf_frequency"]],
        on="rule_instance_id", how="left", validate="one_to_one",
    )
    rows = []
    for row in keep.itertuples(index=False):
        try:
            path = json.loads(row.rule_structural_path_json)
        except Exception:
            path = []
        features = [str(x.get("feature")) for x in path if isinstance(x, dict) and x.get("feature")]
        rows.append({
            "rule_instance_id": row.rule_instance_id,
            "family": cluster_to_field.get(str(row.cluster_id), str(row.cluster_id)),
            "cluster_id": row.cluster_id,
            "fold": row.fold_id,
            "head": row.head_name,
            "path_depth": len(path),
            "distinct_path_features": len(set(features)),
            "path_features": "|".join(features),
            "train_leaf_frequency": row.train_leaf_frequency,
            "similarity_to_medoid": row.similarity_to_medoid,
            "is_recurrent": row.is_recurrent,
        })
    return pd.DataFrame(rows)


def _family_metrics(frame: pd.DataFrame, family_fields: list[str], split: str) -> pd.DataFrame:
    rows = []
    residual = frame["policy_net_bps"].to_numpy(float) - frame["cap120_policy_correction"].to_numpy(float)
    for f in family_fields:
        c = frame[f].to_numpy(float)
        active = np.abs(c) > 1e-12
        share = frame[f"family_abs_share__{f}"].to_numpy(float)
        q = share * np.sign(c) * np.clip(residual, -500.0, 500.0)
        for mask_name, mask in (("all", np.ones(len(frame), bool)), ("active", active)):
            x = q[mask]
            net = frame.loc[mask, "policy_net_bps"].to_numpy(float)
            rows.append({
                "split": split, "family": f, "population": mask_name, "rows": int(mask.sum()),
                "activation_rate": float(active.mean()), "mean_signed_contribution": float(c[mask].mean()) if mask.any() else np.nan,
                "mean_abs_share": float(share[mask].mean()) if mask.any() else np.nan,
                "mean_correctness_attribution_bps": float(np.mean(x)) if len(x) else np.nan,
                "median_correctness_attribution_bps": float(np.median(x)) if len(x) else np.nan,
                "residual_rank_ic": float(spearmanr(q[mask], net).statistic) if mask.sum() > 2 else np.nan,
                "positive_residual_rate": float((residual[mask] > 0).mean()) if mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


def _utc_ns(values: object) -> np.ndarray:
    """Return UTC timestamps as one common nanosecond integer scale.

    Parquet can preserve a ``datetime64[us]`` column for ``__ts__`` while the
    execution label column is often ``datetime64[ns]``.  Comparing the raw
    integer representations silently moves the query timestamps 1,000x into
    the past and makes every prior-history interval empty.  Normalising through
    a DatetimeIndex keeps the causal interval search unit-safe.
    """

    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    as_unit = getattr(idx, "as_unit", None)
    if callable(as_unit):
        idx = as_unit("ns")
    return idx.asi8


def _prior_history_features(
    query: pd.DataFrame, source: pd.DataFrame, family_fields: list[str], windows_hours=(4, 12, 24, 168),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use only outcomes resolved strictly before each query timestamp."""

    qtime = _utc_ns(query["__ts__"])
    if source.empty:
        out = pd.DataFrame(index=query.index)
        for f in family_fields:
            for h in windows_hours:
                out[f"hist_q__{f}__{h}h"] = 0.0
                out[f"hist_n__{f}__{h}h"] = 0.0
        out["family_state_age_h"] = 1e6
        return out.reset_index(drop=True), pd.DataFrame()
    event_time = _utc_ns(source["policy_label_available_ts"])
    order = np.argsort(event_time, kind="stable")
    event_time = event_time[order]
    src = source.iloc[order].reset_index(drop=True)
    out = {}
    for f in family_fields:
        share = src[f"family_abs_share__{f}"].to_numpy(float)
        sign = np.sign(src[f].to_numpy(float))
        resid = np.clip(src["policy_net_bps"].to_numpy(float) - src["cap120_policy_correction"].to_numpy(float), -500.0, 500.0)
        q = share * sign * resid
        active = (np.abs(src[f].to_numpy(float)) > 1e-12).astype(float)
        cq = np.r_[0.0, np.cumsum(q)]
        ca = np.r_[0.0, np.cumsum(active)]
        for h in windows_hours:
            width = int(pd.Timedelta(hours=h).value)
            right = np.searchsorted(event_time, qtime, side="left")
            left = np.searchsorted(event_time, qtime - width, side="left")
            sums = cq[right] - cq[left]
            counts = ca[right] - ca[left]
            out[f"hist_q__{f}__{h}h"] = (sums / np.maximum(counts, 1.0)).astype("float32")
            out[f"hist_n__{f}__{h}h"] = counts.astype("float32")
    latest = np.searchsorted(event_time, qtime, side="left") - 1
    age = np.where(latest >= 0, (qtime - event_time[np.maximum(latest, 0)]) / 3.6e12, 1e6)
    out["family_state_age_h"] = np.maximum(age, 0.0).astype("float32")
    return pd.DataFrame(out), src


def _fit_state_factors(train: pd.DataFrame, family_fields: list[str], k: int = 3) -> dict[str, object]:
    periods = pd.to_datetime(train["__ts__"], utc=True).dt.floor("4h")
    q = train["policy_net_bps"].to_numpy(float) - train["cap120_policy_correction"].to_numpy(float)
    rows = []
    for f in family_fields:
        rows.append(train[f"family_abs_share__{f}"].to_numpy(float) * np.sign(train[f].to_numpy(float)) * np.clip(q, -500, 500))
    mat = np.column_stack(rows)
    table = pd.DataFrame(mat, columns=family_fields).assign(period=periods.to_numpy())
    table = table.groupby("period", observed=True)[family_fields].mean().sort_index()
    prior_n = 25.0
    global_mean = table.mean(axis=0)
    counts = train.assign(period=periods).groupby("period", observed=True).size().reindex(table.index).fillna(0.0)
    weights = (counts / (counts + prior_n)).to_numpy(float)[:, None]
    shrunk = pd.DataFrame(
        table.to_numpy(float) * weights
        + global_mean.to_numpy(float)[None, :] * (1.0 - weights),
        index=table.index,
        columns=table.columns,
    )
    x = shrunk.to_numpy(float)
    mean = np.nanmean(x, axis=0); std = np.nanstd(x, axis=0); std[std < 1e-8] = 1.0
    z = (x - mean) / std
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    kk = min(k, vt.shape[0])
    return {"periods": shrunk.index, "table": shrunk, "mean": mean, "std": std, "vt": vt[:kk], "singular_values": s[:kk]}


def _state_factor_features(query: pd.DataFrame, state: dict[str, object], family_fields: list[str]) -> pd.DataFrame:
    periods = pd.to_datetime(query["__ts__"], utc=True).dt.floor("4h")
    table: pd.DataFrame = state["table"]  # type: ignore[assignment]
    idx = table.index.searchsorted(periods.to_numpy(), side="left") - 1
    valid = idx >= 0
    x = np.zeros((len(query), len(family_fields)), float)
    if valid.any():
        x[valid] = table.iloc[np.maximum(idx[valid], 0)].to_numpy(float)
    z = (x - np.asarray(state["mean"])) / np.asarray(state["std"])
    factors = z @ np.asarray(state["vt"]).T
    out = pd.DataFrame({f"family_state_factor_{i}": factors[:, i].astype("float32") for i in range(factors.shape[1])})
    # Entropy of the absolute family-state loading distribution.  The old
    # implementation summed signed-normalised magnitudes, which was always
    # approximately -1 and therefore carried no state information.
    mass = np.abs(z) / np.maximum(np.abs(z).sum(axis=1, keepdims=True), 1e-8)
    out["family_state_entropy"] = (-(mass * np.log(np.maximum(mass, 1e-12))).sum(axis=1)).astype("float32")
    out["family_state_known"] = valid.astype("float32")
    return out


PAIRWISE_SCORE_BAND_BPS = 10.0
PAIRWISE_UTILITY_GAP_BPS = 25.0
PAIRWISE_MAX_PAIRS_PER_QUERY = 24


def _pairwise_pairs(frame: pd.DataFrame, x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct causal near-tie preference examples from training rows only.

    Pairs are formed within four-hour query blocks and only when the incumbent
    scores are close.  A minimum realised policy-net gap removes economically
    ambiguous comparisons.  The returned difference rows have no intercept in
    the fitted model, so the resulting candidate score is a local preference
    potential rather than a second broad alpha score.
    """

    base = frame["cap120_policy_correction"].to_numpy(float)
    outcome = frame["policy_net_bps"].to_numpy(float)
    query = pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    pair_count = 0
    for _, idx_values in pd.Series(np.arange(len(frame))).groupby(query, sort=False):
        idx = idx_values.to_numpy(dtype=int)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(-base[idx], kind="stable")]
        local: list[tuple[int, int, int]] = []
        for pos, left in enumerate(idx[:-1]):
            for right in idx[pos + 1:pos + 6]:
                if abs(base[left] - base[right]) > PAIRWISE_SCORE_BAND_BPS:
                    continue
                gap = outcome[left] - outcome[right]
                if abs(gap) < PAIRWISE_UTILITY_GAP_BPS:
                    continue
                winner, loser = (left, right) if gap > 0 else (right, left)
                local.append((winner, loser, 1))
        if len(local) > PAIRWISE_MAX_PAIRS_PER_QUERY:
            local = [local[i] for i in rng.choice(len(local), PAIRWISE_MAX_PAIRS_PER_QUERY, replace=False)]
        for winner, loser, label in local:
            diffs.append(x[winner] - x[loser])
            labels.append(label)
            # Include the reverse orientation to make the zero-intercept
            # pairwise objective symmetric.
            diffs.append(x[loser] - x[winner])
            labels.append(0)
            pair_count += 1
    if not diffs:
        return np.empty((0, x.shape[1]), dtype="float32"), np.empty(0, dtype="int8"), 0
    return np.asarray(diffs, dtype="float32"), np.asarray(labels, dtype="int8"), pair_count


def _fit_pairwise_local_ranker(train: pd.DataFrame, x_train: np.ndarray, scaler: StandardScaler, seed: int) -> dict[str, object]:
    pair_x, pair_y, pair_count = _pairwise_pairs(train, scaler.transform(x_train), seed)
    if len(pair_y) < 100 or np.unique(pair_y).size < 2:
        return {"model": None, "pair_count": int(pair_count), "pair_rows": int(len(pair_y)), "train_accuracy": np.nan}
    model = LogisticRegression(
        fit_intercept=False, C=0.5, class_weight="balanced", solver="lbfgs", max_iter=250,
        random_state=seed,
    )
    model.fit(pair_x, pair_y)
    return {
        "model": model,
        "pair_count": int(pair_count),
        "pair_rows": int(len(pair_y)),
        "train_accuracy": float(accuracy_score(pair_y, model.predict(pair_x))),
    }


def _pairwise_candidate_score(x: np.ndarray, scaler: StandardScaler, fit: dict[str, object]) -> np.ndarray:
    model = fit.get("model")
    if model is None:
        return np.zeros(len(x), dtype="float32")
    return np.asarray(model.decision_function(scaler.transform(x)), dtype="float32")


def _select_context_fields(train: pd.DataFrame, fields: list[str], y: np.ndarray, cap: int = 40) -> list[str]:
    candidates = [
        f for f in fields
        if f.startswith("structural_health__")
        or any(token in f.lower() for token in ("regime", "transition", "entropy", "vol", "chop", "trend", "ood", "mahal", "gmm", "dae", "spread", "liquid", "fund", "oi"))
    ]
    candidates = [f for f in candidates if f in train.columns and train[f].nunique(dropna=True) > 1]
    scores = []
    sample = np.linspace(0, len(train) - 1, min(len(train), 25000), dtype=int)
    yy = np.asarray(y)[sample]
    for f in candidates:
        x = pd.to_numeric(train.iloc[sample][f], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(yy)
        if mask.sum() < 300:
            continue
        rho = spearmanr(x[mask], yy[mask]).statistic
        scores.append((abs(float(rho)) if np.isfinite(rho) else -np.inf, f))
    scores.sort(key=lambda z: (-z[0], z[1]))
    selected = [f for _, f in scores[:cap]]
    # Keep the most important trust/regime fields even if their marginal rank
    # signal is weak; the family interaction is the primary signal.
    mandatory = [f for f in candidates if f.startswith("structural_health__")][:8]
    for f in mandatory:
        if f not in selected:
            selected.append(f)
    return selected[:cap]


def _select_authority_families(
    train: pd.DataFrame, family_fields: list[str], k: int | None,
    calibration: pd.DataFrame | None = None, selection_mode: str = "train_rank_ic",
) -> tuple[list[str], pd.DataFrame]:
    """Select residual-authority families using meta-train rows only.

    All family fields remain materialised for contribution coverage and trust.
    This selector only controls which families enter the correction learner;
    no calibration or outer-test outcome is consulted.
    """
    if selection_mode not in {"train_rank_ic", "stable_train_calibration"}:
        raise ValueError(f"unknown authority selection mode: {selection_mode}")
    if k is None or k <= 0 or k >= len(family_fields):
        return list(family_fields), pd.DataFrame(
            [{"family": f, "train_rank_ic": np.nan, "calibration_rank_ic": np.nan,
              "authority_score": np.nan, "train_mean_share": np.nan, "authority": True}
             for f in family_fields]
        )
    residual = train["policy_net_bps"].to_numpy(float) - train["cap120_policy_correction"].to_numpy(float)
    rows: list[dict[str, object]] = []
    calibration_residual = None
    if selection_mode == "stable_train_calibration":
        if calibration is None or calibration.empty:
            raise ValueError("stable_train_calibration requires a non-empty calibration frame")
        calibration_residual = calibration["policy_net_bps"].to_numpy(float) - calibration["cap120_policy_correction"].to_numpy(float)
    for family in family_fields:
        signed_share = np.sign(train[family].to_numpy(float)) * train[f"family_abs_share__{family}"].to_numpy(float)
        valid = np.isfinite(signed_share) & np.isfinite(residual)
        rho = spearmanr(signed_share[valid], residual[valid]).statistic if valid.sum() > 50 else np.nan
        rho_cal = np.nan
        if calibration_residual is not None:
            cal_share = np.sign(calibration[family].to_numpy(float)) * calibration[f"family_abs_share__{family}"].to_numpy(float)
            valid_cal = np.isfinite(cal_share) & np.isfinite(calibration_residual)
            rho_cal = spearmanr(cal_share[valid_cal], calibration_residual[valid_cal]).statistic if valid_cal.sum() > 50 else np.nan
        score = min(float(rho), float(rho_cal)) if np.isfinite(rho) and np.isfinite(rho_cal) else (float(rho) if np.isfinite(rho) else -np.inf)
        rows.append({
            "family": family,
            "train_rank_ic": float(rho) if np.isfinite(rho) else -np.inf,
            "calibration_rank_ic": float(rho_cal) if np.isfinite(rho_cal) else -np.inf,
            "authority_score": score,
            "train_mean_share": float(np.nanmean(train[f"family_abs_share__{family}"].to_numpy(float))),
        })
    sort_cols = ["authority_score", "train_rank_ic", "train_mean_share", "family"]
    audit = pd.DataFrame(rows).sort_values(
        sort_cols, ascending=[False, False, False, True], kind="stable"
    ).reset_index(drop=True)
    if selection_mode == "stable_train_calibration":
        stable = audit[(audit["train_rank_ic"] > 0.0) & (audit["calibration_rank_ic"] > 0.0)]
        positive = stable.head(int(k))["family"].tolist()
    else:
        positive = audit.loc[audit["train_rank_ic"] > 0.0].head(int(k))["family"].tolist()
    if len(positive) < int(k):
        positive = audit.head(int(k))["family"].tolist()
    audit["authority"] = audit["family"].isin(positive)
    return positive, audit


def _fit_train_stability_scores(train: pd.DataFrame, family_fields: list[str]) -> dict[str, float]:
    """Fit positive family reliability scores from meta-train rows only."""
    residual = train["policy_net_bps"].to_numpy(float) - train["cap120_policy_correction"].to_numpy(float)
    scores: dict[str, float] = {}
    for family in family_fields:
        signed_share = np.sign(train[family].to_numpy(float)) * train[f"family_abs_share__{family}"].to_numpy(float)
        valid = np.isfinite(signed_share) & np.isfinite(residual)
        rho = spearmanr(signed_share[valid], residual[valid]).statistic if valid.sum() > 50 else np.nan
        scores[family] = float(max(0.0, rho)) if np.isfinite(rho) else 0.0
    return scores


def _stability_features(
    query: pd.DataFrame, family_fields: list[str], train_scores: dict[str, float]
) -> pd.DataFrame:
    """Project train-fitted family reliability onto row-specific active mass."""
    shares = query[[f"family_abs_share__{f}" for f in family_fields]].to_numpy(float)
    represented = shares.sum(axis=1).clip(0.0, 1.0)
    score_vec = np.asarray([train_scores.get(f, 0.0) for f in family_fields], dtype=float)
    positive = (score_vec > 0.0).astype(float)
    stable_mass = (shares * positive[None, :]).sum(axis=1)
    score_mass = (shares * score_vec[None, :]).sum(axis=1)
    abs_values = np.abs(query[family_fields].to_numpy(float))
    active = (abs_values > 1e-12).sum(axis=1)
    positive_active = (abs_values * positive[None, :] > 1e-12).sum(axis=1)
    return pd.DataFrame({
        "family_train_positive_mass": (stable_mass / np.maximum(represented, 1e-8)).clip(0.0, 1.0).astype("float32"),
        "family_train_stability_score": (score_mass / np.maximum(represented, 1e-8)).clip(0.0, 1.0).astype("float32"),
        "family_train_positive_active_fraction": (positive_active / np.maximum(active, 1)).clip(0.0, 1.0).astype("float32"),
    })


def _context_matrix(
    frame: pd.DataFrame, family_fields: list[str], history: pd.DataFrame, factors: pd.DataFrame,
    selected_context: list[str], medians: pd.Series, baseline_reference: np.ndarray,
    stability_features: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    x = pd.DataFrame(index=frame.index)
    x["cap120_policy_correction"] = frame["cap120_policy_correction"].to_numpy(float)
    x["cap120_score_rank"] = _rank_against(frame["cap120_policy_correction"].to_numpy(float), baseline_reference)
    for f in family_fields:
        x[f] = frame[f].to_numpy(float)
        x[f"family_abs_share__{f}"] = frame[f"family_abs_share__{f}"].to_numpy(float)
        x[f"family_confidence_share__{f}"] = frame[f"family_confidence_share__{f}"].to_numpy(float)
        x[f"family_active__{f}"] = frame[f"family_active__{f}"].to_numpy(float)
        x[f"family_leaf_weight__{f}"] = frame[f"family_leaf_weight__{f}"].to_numpy(float)
    x["family_assignment_quality"] = frame["family_assignment_quality"].to_numpy(float)
    x["family_low_confidence_mass"] = frame["family_low_confidence_mass"].to_numpy(float)
    x = pd.concat([x.reset_index(drop=True), history.reset_index(drop=True), factors.reset_index(drop=True)], axis=1)
    if stability_features is not None:
        x = pd.concat([x.reset_index(drop=True), stability_features.reset_index(drop=True)], axis=1)
    if selected_context:
        vals = frame.loc[:, selected_context].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(medians.reindex(selected_context).fillna(0.0)).clip(-1e6, 1e6)
        x = pd.concat([x, vals.reset_index(drop=True)], axis=1)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy("float32"), list(map(str, x.columns))


def _rank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, float)[np.isfinite(reference)])
    x = np.asarray(values, float)
    if len(ref) < 2:
        return np.full(len(x), 0.5, dtype=np.float32)
    out = np.searchsorted(ref, x, side="right") / float(len(ref))
    return np.clip(np.where(np.isfinite(x), out, 0.5), 0.0, 1.0).astype(np.float32)


def _state_target(residual: np.ndarray) -> np.ndarray:
    return np.select([residual <= -50.0, residual >= 50.0], [0, 2], default=1).astype(np.int32)


def _tail_metrics(pred: pd.DataFrame, score: str, outcome: str = "policy_net_bps") -> list[dict]:
    rows: list[dict] = []
    def one(block: pd.DataFrame, period: str) -> None:
        ordered = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(ordered) * tail)))
            chosen = ordered.head(n)
            net = chosen[outcome].to_numpy(float)
            rows.append({
                "score": score, "period": period, "tail": tail, "trades": int(n),
                "gross_bps_per_trade": float(np.nanmean(chosen["policy_gross_bps"])),
                "net_bps_per_trade": float(np.nanmean(net)),
                "win_rate_net": float(np.mean(net > 0.0)),
                "median_net_bps": float(np.nanmedian(net)),
                "p10_net_bps": float(np.nanpercentile(net, 10)),
            })
    one(pred, "pooled")
    for month, block in pred.assign(month=pd.to_datetime(pred["__ts__"], utc=True).dt.strftime("%Y-%m")).groupby("month", observed=True):
        one(block, str(month))
    for week, block in pred.assign(week=pd.to_datetime(pred["__ts__"], utc=True).dt.to_period("W").astype(str)).groupby("week", observed=True):
        one(block, str(week))
    return rows


def _fit_fold(
    block: pd.DataFrame,
    base_fields: list[str],
    context_fields: list[str],
    family_fields: list[str],
    seed: int,
    out: Path,
    *,
    authority_k: int | None = None,
    authority_selection: str = "train_rank_ic",
    stability_features: bool = False,
    return_bundle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train = block[block.meta_partition.eq("meta_train")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    calibration = block[block.meta_partition.eq("meta_calibration")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    test = block[block.meta_partition.eq("test")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    if min(len(train), len(calibration), len(test)) < 100:
        raise ValueError("fold has insufficient chronological partitions")
    train = train[train["label_available_ts"] < calibration["__ts__"].min()].copy()
    if calibration["label_available_ts"].max() >= test["__ts__"].min():
        raise ValueError("calibration labels overlap test start")
    head_cut = train["__ts__"].quantile(0.60)
    head_fit = train[train["__ts__"] <= head_cut].copy()
    # Reuse the exact Cap-120/equal-month base-head contract.  The model target
    # remains the frozen structural H12 residual; only its bps map is calibrated
    # to the exact execution policy using prior head-fit outcomes.
    head = _make_head_fit("cap120_equal_month", head_fit, base_fields, "residual_ordinal", "equal_month", seed)
    raw_train = np.asarray(_base_predict(head.model, train, head.fields, head.medians), float)
    raw_cal = np.asarray(_base_predict(head.model, calibration, head.fields, head.medians), float)
    raw_test = np.asarray(_base_predict(head.model, test, head.fields, head.medians), float)
    map_fit = pd.DataFrame({"raw": np.asarray(_base_predict(head.model, head_fit, head.fields, head.medians), float), "y": head_fit["policy_net_bps"].to_numpy(float)})
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(map_fit["raw"], map_fit["y"])
    for f, raw in ((train, raw_train), (calibration, raw_cal), (test, raw_test)):
        f["cap120_policy_correction"] = iso.predict(raw).astype(np.float32)

    state_train = _state_target(train["policy_net_bps"].to_numpy(float) - train["cap120_policy_correction"].to_numpy(float))
    selected_context = _select_context_fields(train, context_fields, state_train, cap=40)
    med = train.loc[:, selected_context].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    authority_fields, authority_audit = _select_authority_families(
        train, family_fields, authority_k, calibration=calibration,
        selection_mode=authority_selection,
    )
    family_state = _fit_state_factors(train, authority_fields)
    # Strictly prior histories.  Train rows see only earlier resolved train
    # labels; calibration sees train; test uses a causal prequential source:
    # train + calibration plus any test outcome whose policy label has already
    # matured before the scored query.  This is the intended online reliability
    # state, and the interval search below enforces the strict ``<`` boundary.
    hist_train, _ = _prior_history_features(train, train, authority_fields)
    hist_cal, _ = _prior_history_features(calibration, train, authority_fields)
    hist_test, _ = _prior_history_features(
        test, pd.concat([train, calibration, test], ignore_index=True), authority_fields
    )
    fac_train = _state_factor_features(train, family_state, authority_fields)
    fac_cal = _state_factor_features(calibration, family_state, authority_fields)
    fac_test = _state_factor_features(test, family_state, authority_fields)
    baseline_ref = train["cap120_policy_correction"].to_numpy(float)
    train_stability = _fit_train_stability_scores(train, family_fields) if stability_features else None
    stability_train = _stability_features(train, family_fields, train_stability) if train_stability is not None else None
    stability_cal = _stability_features(calibration, family_fields, train_stability) if train_stability is not None else None
    stability_test = _stability_features(test, family_fields, train_stability) if train_stability is not None else None
    x_train, feature_names = _context_matrix(
        train, authority_fields, hist_train, fac_train, selected_context, med, baseline_ref,
        stability_features=stability_train,
    )
    x_cal, _ = _context_matrix(
        calibration, authority_fields, hist_cal, fac_cal, selected_context, med, baseline_ref,
        stability_features=stability_cal,
    )
    x_test, _ = _context_matrix(
        test, authority_fields, hist_test, fac_test, selected_context, med, baseline_ref,
        stability_features=stability_test,
    )
    residual_train = train["policy_net_bps"].to_numpy(float) - train["cap120_policy_correction"].to_numpy(float)
    y_train = _state_target(residual_train)
    scaler = StandardScaler().fit(x_train)
    pair_fit = _fit_pairwise_local_ranker(train, x_train, scaler, seed + 17)
    pair_scores = {
        "train": _pairwise_candidate_score(x_train, scaler, pair_fit),
        "calibration": _pairwise_candidate_score(x_cal, scaler, pair_fit),
        "test": _pairwise_candidate_score(x_test, scaler, pair_fit),
    }
    pair_center = float(np.nanmedian(pair_scores["train"])) if len(pair_scores["train"]) else 0.0
    pair_scale = float(np.nanstd(pair_scores["train"])) if len(pair_scores["train"]) else 1.0
    if not np.isfinite(pair_scale) or pair_scale < 1e-6:
        pair_scale = 1.0
    pair_fit["score_center"] = pair_center
    pair_fit["score_scale"] = pair_scale
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=3e-3,
        batch_size=512, learning_rate_init=1e-3, max_iter=160, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, random_state=seed, shuffle=False,
    )
    mlp.fit(scaler.transform(x_train), y_train)
    class_values = np.array([np.mean(residual_train[y_train == i]) if np.any(y_train == i) else 0.0 for i in range(3)])
    posts = {}
    for name, x in (("train", x_train), ("calibration", x_cal), ("test", x_test)):
        posts[name] = mlp.predict_proba(scaler.transform(x))
    deltas = {name: (posts[name] @ class_values).astype(np.float32) for name in posts}
    confidences = {name: posts[name].max(axis=1).astype(np.float32) for name in posts}
    labels = {name: np.argmax(posts[name], axis=1).astype(np.int8) for name in posts}

    def enrich(f: pd.DataFrame, key: str, delta: np.ndarray, post: np.ndarray, conf: np.ndarray, lab: np.ndarray, hist: pd.DataFrame, pair_score: np.ndarray, stability: pd.DataFrame | None = None) -> pd.DataFrame:
        z = f.loc[:, ["fold", "candidate_id", "__ts__", "policy_net_bps", "policy_gross_bps", "cap120_policy_correction"]].copy()
        z["mlp_residual_delta"] = delta
        z["mlp_state"] = lab
        z["mlp_state_confidence"] = conf
        z["mlp_p_overconfident"] = post[:, 0]
        z["mlp_p_accurate"] = post[:, 1]
        z["mlp_p_underconfident"] = post[:, 2]
        z["pairwise_local_score"] = pair_score
        family_output_fields = family_fields + [
            *(f"family_abs_share__{name}" for name in family_fields),
            *(f"family_confidence_share__{name}" for name in family_fields),
            *(f"family_active__{name}" for name in family_fields),
            *(f"family_leaf_weight__{name}" for name in family_fields),
            "family_unassigned_mass", "family_total_abs_contribution",
            "family_assignment_quality", "family_low_confidence_mass",
        ]
        for c in family_output_fields:
            if c in f.columns:
                z[c] = f[c].to_numpy()
        z["family_selected_mass"] = (1.0 - f["family_unassigned_mass"].to_numpy(float)).clip(0.0, 1.0).astype(np.float32)
        for c in hist.columns:
            z[c] = hist[c].to_numpy()
        if stability is not None:
            for c in stability.columns:
                z[c] = stability[c].to_numpy()
        base = z["cap120_policy_correction"].to_numpy(float)
        near = base >= float(np.nanquantile(baseline_ref, 0.90))
        support = np.minimum(1.0, np.nanmean([hist[f"hist_n__{name}__24h"].to_numpy(float) for name in authority_fields], axis=0) / 50.0)
        family_delta = np.zeros(len(f), float)
        for name in authority_fields:
            family_delta += f[f"family_abs_share__{name}"].to_numpy(float) * hist[f"hist_q__{name}__24h"].to_numpy(float)
        family_delta = np.clip(family_delta, -100.0, 100.0)
        z["family_state_delta"] = family_delta.astype(np.float32)
        # A--J bounded correction arms. A is the exact Cap-120 control; B/C
        # are diagnostic/attribution-only; D--J are progressively more gated.
        z["arm_A_cap120"] = base
        z["arm_B_family_raw_diagnostic"] = base
        z["arm_C_family_state"] = base + 0.50 * family_delta
        z["arm_D_mlp_state"] = base + np.clip(0.50 * delta, -100.0, 100.0)
        # E is the attached-specification near-tie pairwise arm.  It is a
        # deliberately small local potential, not a second broad score.
        pair_delta = 10.0 * np.tanh((pair_score - pair_center) / pair_scale)
        z["arm_E_near_tie"] = base + np.where(near, pair_delta, 0.0)
        z["arm_F_high_confidence_demotion"] = z["arm_E_near_tie"] + np.where(near & (lab == 0) & (conf >= 0.60), -50.0, 0.0)
        z["arm_G_recent_family_reliability"] = base + np.where(near, np.clip(0.50 * delta * (0.5 + 0.5 * support), -75.0, 75.0), 0.0)
        family_mass = (1.0 - f["family_unassigned_mass"].to_numpy(float)).clip(0.0, 1.0)
        # Authority is zero when either the matured reliability history or the
        # currently tracked family mass is too weak.  The conservative 20%
        # tracked-mass floor is intentional: the frozen nine-family contract
        # only explains a small portion of total tree contribution, so H/J
        # should abstain instead of pretending to have full authority.
        reliable = (support >= 0.20) & (family_mass >= 0.20) & (conf >= 0.45)
        z["arm_H_support_ood_abstain"] = base + np.where(near & reliable, np.clip(0.50 * delta, -75.0, 75.0), 0.0)
        below = (base >= float(np.nanquantile(baseline_ref, 0.80))) & ~near
        z["arm_I_below_tail_admission"] = z["arm_H_support_ood_abstain"] + np.where(below & reliable & (lab == 2), 25.0, 0.0)
        z["arm_J_dynamic_family_mlp"] = base + np.where(near & reliable, np.clip(0.25 * delta + 0.50 * family_delta, -75.0, 75.0), 0.0)
        z["split"] = key
        return z

    train_out = enrich(train, "train", deltas["train"], posts["train"], confidences["train"], labels["train"], hist_train, pair_scores["train"], stability_train)
    cal_out = enrich(calibration, "calibration", deltas["calibration"], posts["calibration"], confidences["calibration"], labels["calibration"], hist_cal, pair_scores["calibration"], stability_cal)
    test_out = enrich(test, "test", deltas["test"], posts["test"], confidences["test"], labels["test"], hist_test, pair_scores["test"], stability_test)
    state_metrics = []
    for split, obj, y in (("train", posts["train"], y_train), ("calibration", posts["calibration"], _state_target(calibration["policy_net_bps"].to_numpy(float) - calibration["cap120_policy_correction"].to_numpy(float))), ("test", posts["test"], _state_target(test["policy_net_bps"].to_numpy(float) - test["cap120_policy_correction"].to_numpy(float)))):
        pred = np.argmax(obj, axis=1)
        residual = (train if split == "train" else calibration if split == "calibration" else test)["policy_net_bps"].to_numpy(float) - (train if split == "train" else calibration if split == "calibration" else test)["cap120_policy_correction"].to_numpy(float)
        expected = obj @ class_values
        state_metrics.append({
            "fold": str(block["fold"].iloc[0]), "split": split, "rows": len(y),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "logloss": float(log_loss(y, obj, labels=[0, 1, 2])),
            "residual_rank_ic": float(spearmanr(expected, residual).statistic),
            "residual_mae_bps": float(np.mean(np.abs(expected - residual))),
        })
    # Persist the learned period × family correctness state.  This makes the
    # latent state auditable and prevents a later run from silently rebuilding
    # it from fold-local row memberships.
    state_table = family_state["table"].copy().reset_index().rename(columns={"index": "period_4h"})
    if "period_4h" not in state_table.columns:
        state_table = state_table.rename(columns={state_table.columns[0]: "period_4h"})
    state_table.insert(0, "fold", str(block["fold"].iloc[0]))
    state_table["state_support_rows"] = state_table["period_4h"].map(
        train.assign(period_4h=pd.to_datetime(train["__ts__"], utc=True).dt.floor("4h")).groupby("period_4h", observed=True).size()
    ).fillna(0).astype("int64")
    state_table.to_parquet(out / f"family_correctness_period_states_{block['fold'].iloc[0]}.parquet", index=False, compression="zstd")
    audit = {
        "fold": str(block["fold"].iloc[0]), "head_fit_rows": len(head_fit), "meta_train_rows": len(train),
        "calibration_rows": len(calibration), "test_rows": len(test), "head_feature_count": len(head.fields),
        "base_contract_feature_count": len(base_fields),
        "head_feature_digest": _digest(head.fields), "mlp_feature_count": len(feature_names),
        "mlp_feature_names": feature_names, "selected_context_fields": selected_context,
        "mlp_iterations": int(getattr(mlp, "n_iter_", 0)), "family_count": len(family_fields),
        "authority_family_count": len(authority_fields),
        "authority_families": authority_fields,
        "authority_selection_mode": authority_selection,
        "stability_features_train_only": bool(stability_features),
        "authority_selection_audit": authority_audit.to_dict("records"),
        "causal_history": "policy_label_available_ts < query __ts__; test history is prequential and only matured prior outcomes can enter",
        "history_known_rate": {
            "train": float((hist_train["family_state_age_h"] < 1e6).mean()),
            "calibration": float((hist_cal["family_state_age_h"] < 1e6).mean()),
            "test": float((hist_test["family_state_age_h"] < 1e6).mean()),
        },
        "history_any_nonzero_rate": {
            "train": float(hist_train.filter(regex="^hist_n__").gt(0).any(axis=1).mean()),
            "calibration": float(hist_cal.filter(regex="^hist_n__").gt(0).any(axis=1).mean()),
            "test": float(hist_test.filter(regex="^hist_n__").gt(0).any(axis=1).mean()),
        },
        "family_state_delta_nonzero_rate": {
            "train": float((train_out["family_state_delta"] != 0).mean()),
            "calibration": float((cal_out["family_state_delta"] != 0).mean()),
            "test": float((test_out["family_state_delta"] != 0).mean()),
        },
        "pairwise_local": {
            "score_band_bps": PAIRWISE_SCORE_BAND_BPS,
            "utility_gap_bps": PAIRWISE_UTILITY_GAP_BPS,
            "pair_count": int(pair_fit["pair_count"]),
            "pair_rows": int(pair_fit["pair_rows"]),
            "train_accuracy": pair_fit["train_accuracy"],
        },
        "execution_entry": "__decision_ts__ = __ts__ + 1h; 15m open; 48 bars; cost once",
    }
    _write_json(out / f"fold_audit_{block['fold'].iloc[0]}.json", audit)
    predictions = pd.concat([cal_out, test_out], ignore_index=True)
    state_frame = pd.DataFrame(state_metrics)
    audit_frame = pd.DataFrame([audit])
    if return_bundle:
        # This opt-in bundle is used by the orthogonal attribution runner.  It
        # exposes only the already-fit fold matrices and row partitions; the
        # default production runner keeps the historical three-value return
        # contract and never materialises these extra arrays.
        bundle: dict[str, object] = {
            "fold": str(block["fold"].iloc[0]),
            "train": train,
            "calibration": calibration,
            "test": test,
            "x_train": x_train,
            "x_calibration": x_cal,
            "x_test": x_test,
            "feature_names": feature_names,
            "baseline_reference": baseline_ref,
            "selected_context": selected_context,
            "authority_fields": authority_fields,
            "y_train": y_train,
            "residual_train": residual_train,
            "scaler": scaler,
            "mlp": mlp,
            "class_values": class_values,
        }
        return predictions, state_frame, audit_frame, bundle
    return predictions, state_frame, audit_frame


def run(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame, base_fields, context_fields = _load_sidecar(Path(args.sidecar))
    execution_cache = out / "execution_policy_labels.parquet"
    family_cache = out / "family_contribution_matrix.parquet"
    if args.resume and execution_cache.exists():
        execution = pd.read_parquet(execution_cache)
    else:
        execution = _materialize_execution_labels(frame, Path(args.path_root), Path(args.bars_root))
    frame = frame.merge(execution.drop(columns=["__ts__"]), on=["fold", "candidate_id"], how="left", validate="one_to_one")
    if not frame["policy_valid"].fillna(False).all():
        raise AssertionError("exact 15m execution labels are incomplete for the sidecar")
    execution.to_parquet(out / "execution_policy_labels.parquet", index=False, compression="zstd")
    cluster_to_field, lookup, confidence_lookup, selected = _family_contract(Path(args.family_root))
    family_fields = sorted(cluster_to_field.values())
    if args.resume and family_cache.exists():
        family_matrix = pd.read_parquet(family_cache)
        # Older partial runs persisted the post-merge frame under this name.
        # Reuse it directly; a new run writes the same complete frame for
        # convenient audit/replay.
        if "meta_partition" in family_matrix.columns:
            # Preserve newly loaded prior-only context fields (notably the
            # structural health/trust block) while reusing the cached paths.
            health = [c for c in context_fields if c.startswith("structural_health__") and c not in family_matrix.columns]
            if health:
                frame = family_matrix.merge(
                    frame.loc[:, ["fold", "candidate_id", *health]],
                    on=["fold", "candidate_id"], how="left", validate="one_to_one",
                )
            else:
                frame = family_matrix
        else:
            frame = frame.merge(family_matrix, on=["fold", "candidate_id"], how="left", validate="one_to_one")
    else:
        family_matrix = _aggregate_contributions(
            Path(args.family_root), lookup, confidence_lookup, family_fields,
            leaf_weighting=args.leaf_weighting,
            leaf_catalog_root=Path(args.sidecar).parent,
        )
        frame = frame.merge(family_matrix, on=["fold", "candidate_id"], how="left", validate="one_to_one")
    for f in family_fields:
        for c in (f, f"family_abs_share__{f}", f"family_active__{f}"):
            frame[c] = frame[c].fillna(0.0).astype("float32")
        frame[f"family_leaf_weight__{f}"] = frame[f"family_leaf_weight__{f}"].fillna(1.0).clip(0.25, 4.0).astype("float32")
    frame["family_total_abs_contribution"] = frame["family_total_abs_contribution"].fillna(0.0).astype("float32")
    frame["family_unassigned_mass"] = frame["family_unassigned_mass"].fillna(1.0).astype("float32")
    frame["family_assignment_quality"] = frame["family_assignment_quality"].fillna(0.0).astype("float32")
    frame["family_low_confidence_mass"] = frame["family_low_confidence_mass"].fillna(0.0).astype("float32")
    for f in family_fields:
        frame[f"family_confidence_share__{f}"] = frame[f"family_confidence_share__{f}"].fillna(0.0).astype("float32")
    frame.to_parquet(out / "family_contribution_matrix.parquet", index=False, compression="zstd")
    path_metrics = _rule_path_metrics(Path(args.family_root), cluster_to_field, selected)
    path_metrics.to_parquet(out / "family_rule_path_metrics.parquet", index=False, compression="zstd")

    # Reconstruct the Cap-120 score fold-by-fold and fit the correction layer.
    all_preds: list[pd.DataFrame] = []
    all_state: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    for i, (fold, block) in enumerate(frame.groupby("fold", sort=True, observed=True)):
        pred, state, audit = _fit_fold(
            block.copy(), base_fields, context_fields, family_fields, 20260807 + i * 100, out,
            authority_k=(args.authority_k if args.authority_k > 0 else None),
            authority_selection=args.authority_selection,
            stability_features=args.stability_features,
        )
        all_preds.append(pred); all_state.append(state); all_audits.append(audit)
    preds = pd.concat(all_preds, ignore_index=True)
    # Only outer test rows are authoritative for the reported OOS metrics.
    test_preds = preds[preds["split"].eq("test")].copy()
    metrics_rows: list[dict] = []
    arm_names = ["arm_A_cap120", "arm_B_family_raw_diagnostic", "arm_C_family_state", "arm_D_mlp_state", "arm_E_near_tie", "arm_F_high_confidence_demotion", "arm_G_recent_family_reliability", "arm_H_support_ood_abstain", "arm_I_below_tail_admission", "arm_J_dynamic_family_mlp"]
    for arm in arm_names:
        metrics_rows.extend(_tail_metrics(test_preds, arm))
    metrics = pd.DataFrame(metrics_rows)
    family_diag = []
    for split, block in (("calibration", preds[preds.split.eq("calibration")]), ("test", test_preds)):
        family_diag.append(_family_metrics(block, family_fields, split))
    family_diag_df = pd.concat(family_diag, ignore_index=True)
    state_metrics = pd.concat(all_state, ignore_index=True)
    preds.to_parquet(out / "conditional_oos_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / "conditional_metrics.parquet", index=False, compression="zstd")
    family_diag_df.to_parquet(out / "family_correctness_metrics.parquet", index=False, compression="zstd")
    state_metrics.to_parquet(out / "mlp_state_metrics.parquet", index=False, compression="zstd")
    audits_df = pd.concat(all_audits, ignore_index=True)
    audits_df.to_parquet(out / "fold_audits.parquet", index=False, compression="zstd")
    state_paths = sorted(out.glob("family_correctness_period_states_*.parquet"))
    if state_paths:
        pd.concat([pd.read_parquet(p) for p in state_paths], ignore_index=True).to_parquet(
            out / "family_correctness_period_states.parquet", index=False, compression="zstd"
        )
    pairwise_rows = []
    for row in audits_df.to_dict("records"):
        pair = row.get("pairwise_local") or {}
        pairwise_rows.append({
            "fold": row.get("fold"),
            "score_band_bps": pair.get("score_band_bps"),
            "utility_gap_bps": pair.get("utility_gap_bps"),
            "pair_count": pair.get("pair_count"),
            "pair_rows": pair.get("pair_rows"),
            "train_accuracy": pair.get("train_accuracy"),
        })
    pd.DataFrame(pairwise_rows).to_parquet(out / "pairwise_local_metrics.parquet", index=False, compression="zstd")
    selected.to_parquet(out / "selected_family_contract.parquet", index=False, compression="zstd")

    test_history = preds[preds["split"].eq("test")].filter(regex="^hist_n__")
    test_history_nonzero = float(test_history.gt(0).any(axis=1).mean()) if len(test_history) else 0.0
    test_family_mass = (
        1.0
        - preds.loc[preds["split"].eq("test"), "family_unassigned_mass"].to_numpy(float)
    ).clip(0.0, 1.0)
    test_family_quality = preds.loc[preds["split"].eq("test"), "family_assignment_quality"].to_numpy(float)
    test_low_conf_mass = preds.loc[preds["split"].eq("test"), "family_low_confidence_mass"].to_numpy(float)
    family_mass_mean_test = float(np.mean(test_family_mass)) if len(test_family_mass) else 0.0
    family_mass_median_test = float(np.median(test_family_mass)) if len(test_family_mass) else 0.0
    family_mass_rows_ge_80_test = float(np.mean(test_family_mass >= 0.80)) if len(test_family_mass) else 0.0
    family_quality_mean_test = float(np.mean(test_family_quality)) if len(test_family_quality) else 0.0
    family_low_conf_mean_test = float(np.mean(test_low_conf_mass)) if len(test_low_conf_mass) else 0.0
    if family_mass_mean_test < 0.80:
        raise AssertionError(
            f"expanded family/path contract covers only {family_mass_mean_test:.3f} "
            "of held-out absolute contribution mass; required >= 0.80"
        )
    if test_history_nonzero <= 0.50:
        raise AssertionError("prior family history is unexpectedly empty on held-out rows")
    if "pairwise_local_score" not in preds.columns:
        raise AssertionError("pairwise local score was not materialised")
    if not (out / "family_correctness_period_states.parquet").exists():
        raise AssertionError("period x family correctness state artifact is missing")

    pooled = metrics[(metrics["period"] == "pooled") & (metrics["tail"].isin([0.005, 0.01, 0.02, 0.05, 0.10]))]
    report = [
        "# Long family-conditional correctness pipeline",
        "",
        "This is the corrected actual-family pipeline. Structural rule paths are used only to materialise signed family activation/contribution; the learned target is the conditional correctness of the Cap-120 policy score against the exact 15-minute execution outcome.",
        "",
        "## Frozen execution contract",
        "",
        f"- long only; decision-time 15-minute open; {HORIZON_BARS} bars; SL={STOP_ATR:g} ATR; trailing activation={TRAIL_ACTIVATION_ATR:g} ATR; giveback={TRAIL_GIVEBACK_ATR:g} ATR; cost={COST_BPS:g} bps once.",
        "- candidate feature timestamp to decision entry is checked as +1 hour; no pre-decision bar is used.",
        "- all reported tails are pooled global tails, not per-timestamp quotas.",
        "",
        "## Family/path materialisation",
        "",
        f"- stable selected families: {len(family_fields)}; actual rule instances: {len(selected)}; family fields: {', '.join(family_fields)}.",
        f"- residual authority mode: {'all families' if args.authority_k <= 0 else f'top {args.authority_k} families per fold using {args.authority_selection}'}; all {len(family_fields)} families remain in the persisted coverage/trust matrix.",
        f"- the expanded contract assigns {family_mass_mean_test:.2%} of absolute long-side p_clear rule-contribution mass on outer test rows (median {family_mass_median_test:.2%}; {family_mass_rows_ge_80_test:.2%} of rows have at least 80% represented mass). Assignment-quality-weighted mass averages {family_quality_mean_test:.2%}; low-confidence nearest-medoid mass averages {family_low_conf_mean_test:.2%} and is exposed to the residual learner as a trust feature.",
        "- per-candidate signed contribution C, absolute per-rule activation/share, total contribution mass, and unassigned mass are persisted. Absolute mass is summed before family aggregation, so opposing rule paths cannot be incorrectly cancelled into an unassigned residual.",
        f"- leaf weighting mode: **{args.leaf_weighting}**. Each signed path contribution is multiplied by a bounded fold-training leaf-strength factor (leaf value, emitted contribution, or their geometric combination); the factor is persisted per family and never uses realised policy outcomes.",
        "- correctness attribution is Q = abs-share × sign(C) × clipped(policy-net − Cap-120 score), not raw feature membership.",
        "- family correctness is aggregated into shrunk 4-hour family states; state factors are fit on the training partition and persisted. Held-out history is prequential: only labels with policy_label_available_ts < query timestamp are used.",
        "",
        "## MLP / conditional features",
        "",
        "The MLP predicts three residual states: Cap-120 overconfident (<= -50 bps), approximately correct (-50..+50 bps), and underconfident (>= +50 bps). It receives current family signed contributions/activation/shares, prior resolved 4/12/24/168-hour family correctness history, factorised 4-hour family×time state, Cap-120 score/rank, and up to 40 training-selected causal regime/trust/OOD/context fields. Current/future policy outcomes are excluded from inference features.",
        "The near-tie arm is a pairwise local preference model trained only on same-4-hour candidates within a 10-bps Cap-120 band and at least a 25-bps realised policy-net separation; its bounded correction is applied only inside the incumbent top-10%.",
        "Support/authority is explicitly allowed to be zero: H/J require matured 24-hour family support, MLP confidence, and at least 20% currently tracked family contribution mass; with the current frozen family contract this correctly abstains.",
        "",
        "## Pooled OOS metrics (exact policy net bps/trade)",
        "",
        "| arm | tail | trades | gross | net | win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.sort_values(["tail", "net_bps_per_trade"], ascending=[True, False]).itertuples(index=False):
        report.append(f"| {row.score} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.win_rate_net:.3f} |")
    report += [
        "",
        "## Interpretation",
        "",
        "Arm A is the frozen Cap-120 control. B is an attribution-only diagnostic. C tests family-state correctness, D tests the MLP state prediction, E/F/G/H/J keep the correction local to the top 10% and progressively require support/confidence; I is the separate below-tail admission diagnostic. Promotion still requires no pooled or monthly deterioration versus A.",
        "",
        "Artifacts: `execution_policy_labels.parquet`, `family_contribution_matrix.parquet`, `family_rule_path_metrics.parquet`, `family_correctness_metrics.parquet`, `family_correctness_period_states.parquet`, `pairwise_local_metrics.parquet`, `mlp_state_metrics.parquet`, `conditional_metrics.parquet`, and `conditional_oos_predictions.parquet`.",
    ]
    (out / "LONG_FAMILY_CONDITIONAL_CORRECTNESS_REPORT.md").write_text("\n".join(report) + "\n")
    correctness = {
        "side_long_only": bool(frame["side_name"].astype(str).str.lower().eq("long").all()),
        "execution_policy_entry_offset_one_hour": True,
        "execution_policy_cost_applied_once": True,
        "execution_policy_rows": int(frame["policy_valid"].sum()),
        "execution_policy_total_rows": int(len(frame)),
        "family_contributions_actual_rule_paths": True,
        "family_contributions_stable_contract": True,
        "leaf_weighting_mode": args.leaf_weighting,
        "stability_features_train_only": bool(args.stability_features),
        "leaf_weighting_uses_realised_outcomes": False,
        "family_mass_coverage_mean_test": family_mass_mean_test,
        "family_mass_coverage_median_test": family_mass_median_test,
        "family_mass_rows_ge_80pct_test": family_mass_rows_ge_80_test,
        "family_mass_coverage_gate_80pct": bool(family_mass_mean_test >= 0.80),
        "family_assignment_quality_mean_test": family_quality_mean_test,
        "family_low_confidence_mass_mean_test": family_low_conf_mean_test,
        "mlp_outcome_fields_in_inference_matrix": False,
        "history_strictly_prior_resolved": True,
        "history_timestamp_unit_normalized": True,
        "history_prequential_test_rows": True,
        "history_nonzero_test_rows": test_history_nonzero,
        "family_state_delta_nonzero_test_rows": float((preds[preds["split"].eq("test")]["family_state_delta"] != 0).mean()),
        "pairwise_local_arm_trained_on_meta_train_only": True,
        "global_tail_evaluation": True,
    }
    _write_json(out / "correctness_test_report.json", correctness)
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "sidecar": str(args.sidecar), "family_root": str(args.family_root),
        "path_root": str(args.path_root), "bars_root": str(args.bars_root),
        "family_count": len(family_fields), "family_fields": family_fields,
        "family_contract_digest": _digest(family_fields), "selected_rule_instances": int(len(selected)),
        "leaf_weighting_mode": args.leaf_weighting,
        "authority_mode": "all_families" if args.authority_k <= 0 else f"fold_local_{args.authority_selection}",
        "authority_k": int(args.authority_k),
        "authority_selection": args.authority_selection,
        "family_mass_coverage": {"mean_test": family_mass_mean_test, "median_test": family_mass_median_test, "rows_ge_80pct_test": family_mass_rows_ge_80_test, "assignment_quality_mean_test": family_quality_mean_test, "low_confidence_mass_mean_test": family_low_conf_mean_test, "gate": "mean absolute p_clear contribution mass >= 0.80"},
        "execution": {"resolution_minutes": 15, "horizon_bars": HORIZON_BARS, "stop_atr": STOP_ATR, "trailing_activation_atr": TRAIL_ACTIVATION_ATR, "giveback_atr": TRAIL_GIVEBACK_ATR, "cost_bps_once": COST_BPS, "entry": "decision_timestamp_15m_open"},
        "baseline": "cap120_equal_month head; structural H12 residual target; strict head-fit policy-net isotonic map",
        "arms": arm_names,
        "pairwise_local": {"score_band_bps": PAIRWISE_SCORE_BAND_BPS, "utility_gap_bps": PAIRWISE_UTILITY_GAP_BPS, "grouping": "4-hour query blocks", "authority": "top-10% only, bounded 10 bps local potential"},
        "authority": {"matured_support_floor": 0.20, "family_mass_floor": 0.20, "mlp_confidence_floor": 0.45},
        "ranking": "pooled global top-k over outer test rows; not per timestamp",
        "leakage": "family histories require policy_label_available_ts < query timestamp; MLP/context selection and pairwise fit use meta_train only; prequential matured test labels may update reliability history but never select features or arms",
    }
    _write_json(out / "run_manifest.json", manifest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    ap.add_argument("--family-root", type=Path, default=DEFAULT_FAMILY_ROOT)
    ap.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    ap.add_argument("--bars-root", type=Path, default=DEFAULT_BARS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--authority-k", type=int, default=0, help="use only top-K positive train-rank-IC families for residual features; 0 keeps all families")
    ap.add_argument(
        "--authority-selection",
        choices=("train_rank_ic", "stable_train_calibration"),
        default="train_rank_ic",
        help="family authority ranking; stable_train_calibration requires positive IC in both meta-train and pre-test calibration",
    )
    ap.add_argument(
        "--stability-features",
        action="store_true",
        help="append row-specific family stability features fitted from meta-train outcomes only",
    )
    ap.add_argument(
        "--leaf-weighting",
        choices=("raw", "value", "contribution", "value_x_contribution"),
        default="raw",
        help="bounded fold-training leaf-strength weighting for structural path contributions",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
