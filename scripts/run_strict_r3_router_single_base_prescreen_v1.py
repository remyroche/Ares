#!/usr/bin/env python3
"""Stage-A router-first single-base screen with strict downstream-safe lineage.

The linked Base funnel begins with four economically related LambdaRank target
geometries.  This runner implements that first screen only:

    frozen full-universe Router -> exact top 50% identities -> one Base head

The Base consumes the selected 72 causal F72 fields and scores *every* routed
candidate.  Router rank is used only to audit the immutable gate; it is not a
feature, output coordinate, score blend, or post-Base cutoff.  Fold-specific
relevance bins for T1--T3 are fitted from strict prior training rows.  Held
scores are persisted target-free before labels are joined for diagnostics.

This is offline research.  It neither refits R/U nor MC1, and it never mutates
the live bundle.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_single_base_prescreen_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
F72_SELECTION = ROOT / "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json"
DEFAULT_HELD_MONTHS = ("2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07")
GAIN_SCHEDULES = {
    # The Stage-A control.  The subsequent funnel alters one loss-geometry
    # element at a time; it does not form a target x gain x objective grid.
    "g1_clipped_economic": [0.0, 0.5, 2.0, 3.0, 6.0, 8.0],
    "g0_soft": [0.0, 0.75, 2.0, 3.0, 5.0, 6.5],
    "g2_stronger_tail": [0.0, 0.5, 2.0, 4.0, 8.0, 12.0],
    "g3_strongest_tail": [0.0, 0.5, 2.0, 5.0, 10.0, 16.0],
}
DEFAULT_GAIN_NAME = "g1_clipped_economic"
PROHIBITED_SCORE_TOKENS = ("policy_", "label_", "target_", "magnitude_", "decision_atr")


@dataclass(frozen=True)
class TargetSpec:
    key: str
    value_column: str
    valid_column: str
    available_column: str
    fixed_ordinal: bool


TARGETS = {
    "t0_policy_ordinal": TargetSpec(
        "t0_policy_ordinal", "policy_ordinal_grade", "policy_ordinal_valid",
        "policy_label_available_ts", True,
    ),
    "t1_raw_bps": TargetSpec(
        "t1_raw_bps", "magnitude_raw_bps", "raw_magnitude_valid",
        "policy_label_available_ts", False,
    ),
    "t2_sqrt_atr": TargetSpec(
        "t2_sqrt_atr", "magnitude_sqrt_atr", "normalised_magnitude_valid",
        "normalised_label_available_ts", False,
    ),
    "t3_atr": TargetSpec(
        "t3_atr", "magnitude_atr", "normalised_magnitude_valid",
        "normalised_label_available_ts", False,
    ),
}


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(
        start.normalize().replace(day=1),
        (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1),
        freq="MS", tz="UTC",
    ))


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    # Folds intentionally overlap in their historical windows.  Deduplicate
    # file paths here so immutable lineage hashing remains complete without
    # repeatedly streaming the same large panel.
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _load_f72_fields(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    fields = payload.get("selected_features")
    # The initial F72 receipt remains the default, but finalist-specific
    # selection may provide an independently sealed compact contract.  The
    # scorer stays bounded at the declared feature-selection ceiling and does
    # not infer a feature set.  The precision/preservation beam deliberately
    # retains 130-field finalists, so a historical 120-field parser must not
    # silently reject their frozen contracts.
    if not isinstance(fields, list) or not 1 <= len(fields) <= 160 or len(set(fields)) != len(fields):
        raise AssertionError(f"{path}: expected a sealed 1..160-field contract")
    if not all(isinstance(item, str) and item for item in fields):
        raise AssertionError(f"{path}: invalid field name")
    return tuple(fields)


def _load_frozen_hpo_config(path: Path | None) -> tuple[dict[str, float | int], dict[str, object] | None]:
    """Load only the tree/regularisation fields from a sealed local-HPO receipt.

    Target geometry, ranking objective, gains, truncation, sigmoid, features,
    router, reserve, and every causal data boundary remain command-line
    contracts.  This deliberately accepts no loss or population override.
    ``min_data_fraction`` is resolved separately for each chronological fold,
    exactly as in the HPO trial, so a fixed config does not silently become a
    different minimum-leaf policy when the available support changes.
    """
    if path is None:
        return {}, None
    payload = json.loads(path.read_text())
    source = payload.get("winner", payload)
    if not isinstance(source, dict):
        raise AssertionError(f"{path}: expected an HPO winner object")
    allowed = {
        "learning_rate", "max_depth", "num_leaves", "min_data_fraction",
        "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2",
        "min_gain_to_split",
    }
    unknown = sorted(set(source).intersection({"objective", "target", "gain_name", "truncation", "sigmoid", "fields"}))
    # A full HPO manifest carries these descriptive locked fields.  They are
    # checked by the caller; they can never override the Base contract.
    contract = {key: payload.get(key) for key in ("target", "objective", "gain_name", "truncation", "sigmoid", "fields") if key in payload}
    config: dict[str, float | int] = {}
    for key in allowed:
        if key in source:
            value = source[key]
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise AssertionError(f"{path}: invalid HPO value for {key}")
            config[key] = int(value) if key in {"max_depth", "num_leaves"} else float(value)
    required = allowed
    if set(config) != required:
        raise AssertionError(f"{path}: expected exactly frozen HPO keys {sorted(required)}, got {sorted(config)}")
    if not 0.0 < float(config["min_data_fraction"]) <= 1.0:
        raise AssertionError(f"{path}: min_data_fraction outside (0, 1]")
    leaf_limit = 2 ** int(config["max_depth"]) - 1
    reconstructed = False
    if int(config["num_leaves"]) > leaf_limit:
        # v1 HPO receipts exposed Optuna's requested leaf count.  The HPO
        # fitter itself always used ``min(requested, 2**depth - 1)``.  Recover
        # that documented fitted value solely for those legacy receipts; v2+
        # receipts must already be explicit and are rejected if inconsistent.
        if "requested_num_leaves" not in source:
            config["num_leaves"] = leaf_limit
            reconstructed = True
        else:
            raise AssertionError(f"{path}: num_leaves exceeds max_depth geometry")
    return config, {
        "path": str(path), "contract": contract, "sha256": _sha256([path]),
        "legacy_effective_leaf_reconstruction": reconstructed,
    }


def _candidate_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}" / "scores_features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _feature_path(root: Path | Sequence[Path], month: pd.Timestamp) -> Path:
    """Resolve exactly one immutable owner for each feature calendar month."""
    roots = (root,) if isinstance(root, Path) else tuple(root)
    paths = [item / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for item in roots]
    existing = [item for item in paths if item.exists()]
    if len(existing) != 1:
        raise AssertionError(
            f"{month:%Y-%m}: expected one feature owner across {[str(item) for item in paths]}, found {len(existing)}"
        )
    return existing[0]


def _label_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}" / "target_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _router_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _top_half_identities(router: pd.DataFrame) -> pd.DataFrame:
    work = router.loc[:, [*IDENTITY, "router_primary_rank"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "router_primary_rank", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    work["__rank__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    work["__size__"] = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    result = work.loc[work["__rank__"].le(np.ceil(work["__size__"].to_numpy(float) * .50)), list(IDENTITY)]
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _rank_desc(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", field, "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    result = np.empty(len(frame), dtype=np.float32)
    result[work["__row__"].to_numpy(np.int64)] = (1.0 - (ordinal - .5) / count).astype(np.float32)
    return result


def _sample_complete_queries(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    work["__month__"] = work["__decision_ts__"].dt.strftime("%Y-%m")
    query = work.loc[:, ["__decision_ts__", "__month__"]].drop_duplicates().copy()
    query["__hash__"] = pd.util.hash_pandas_object(
        query["__decision_ts__"].astype(str) + f"|{SEED}", index=False,
    ).to_numpy(np.uint64)
    month_rows = work.groupby("__month__", sort=False).size()
    selected: list[pd.Timestamp] = []
    for month, queries in query.sort_values(["__month__", "__hash__"], kind="stable").groupby("__month__", sort=False):
        allocation = max(1, int(math.ceil(cap * month_rows[str(month)] / len(work))))
        used = 0
        for stamp in queries["__decision_ts__"]:
            size = int((work["__decision_ts__"] == stamp).sum())
            if used and used + size > allocation:
                break
            selected.append(stamp)
            used += size
    return work.loc[work["__decision_ts__"].isin(selected)].drop(columns=["__month__"], errors="ignore").copy()


def _query_groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)


def _numeric_matrix(frame: pd.DataFrame, fields: Sequence[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    numeric = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = numeric.median(axis=0).fillna(0.0)
    return numeric.fillna(medians).fillna(0.0).to_numpy(np.float32), medians


def _fold_bins(values: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = raw[np.isfinite(raw)]
    if len(valid) < 100:
        raise AssertionError("insufficient valid training target values for fold bins")
    clip_low, clip_high = np.quantile(valid, (.02, .98))
    clipped = np.clip(raw, clip_low, clip_high)
    edges = np.quantile(clipped[np.isfinite(clipped)], (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6))
    # Equal-valued bins can occur in quiet regimes.  Searchsorted remains
    # monotonic and the receipt explicitly records the collapsed boundaries.
    labels = np.searchsorted(edges, clipped, side="right").clip(0, 5).astype(np.int8)
    return labels, {
        "clip_p02": float(clip_low), "clip_p98": float(clip_high),
        "bin_edges": [float(item) for item in edges],
        "collapsed_edge_count": int(np.sum(np.diff(edges) <= 0.0)),
    }


def _target_labels(train: pd.DataFrame, held: pd.DataFrame, spec: TargetSpec) -> tuple[np.ndarray, dict[str, object]]:
    if spec.fixed_ordinal:
        labels = pd.to_numeric(train[spec.value_column], errors="raise").to_numpy(np.int8)
        if ((labels < 0) | (labels > 5)).any():
            raise AssertionError("fixed ordinal target outside 0..5")
        return labels, {"binning": "fixed", "edges_bps": [0, 50, 100, 200, 400]}
    labels, description = _fold_bins(train[spec.value_column])
    description["binning"] = "training_fold_winsor_p02_p98_then_six_quantile_bins"
    description["held_target_transform"] = "same strict-training bin edges; labels never required for held scoring"
    return labels, description


def _join_month(*, candidate_root: Path | None, feature_root: Path | Sequence[Path], label_root: Path,
                router_root: Path, month: pd.Timestamp, fields: Sequence[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_path = _candidate_path(candidate_root, month) if candidate_root is not None else None
    feature_path = _feature_path(feature_root, month)
    label_path = _label_path(label_root, month)
    router_path = _router_path(router_root, month)
    features = pd.read_parquet(feature_path, columns=[*IDENTITY, *fields]).copy()
    labels = pd.read_parquet(label_path).copy()
    router = pd.read_parquet(router_path, columns=[*IDENTITY, "router_primary_rank"]).copy()
    for frame in (features, labels, router):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: duplicate candidate ID")
    expected_ids = _top_half_identities(router)
    if candidate_path is not None:
        candidates = pd.read_parquet(candidate_path, columns=list(IDENTITY)).copy()
        candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
        if candidates["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: duplicate candidate ID")
    else:
        # The direct Router mode is a target-free historical score producer:
        # the exact Router50 identity set is all it imports from the router.
        candidates = expected_ids.copy()
    if not candidates["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: expected long-only candidates")
    source_ids = candidates.loc[:, list(IDENTITY)].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if not source_ids.equals(expected_ids):
        left = source_ids.merge(expected_ids, on=list(IDENTITY), how="outer", indicator=True)
        raise AssertionError(f"{month:%Y-%m}: candidate source is not exact frozen Router-50; {left['_merge'].value_counts().to_dict()}")
    frame = candidates.merge(features, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(frame) != len(candidates):
        raise AssertionError(f"{month:%Y-%m}: causal-feature join changed routed identities")
    if frame.loc[:, list(fields)].columns.tolist() != list(fields):
        raise AssertionError(f"{month:%Y-%m}: sealed feature order changed")
    frame = frame.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(frame) != len(candidates):
        raise AssertionError(f"{month:%Y-%m}: label join changed routed identities")
    for column in ("policy_label_available_ts", "normalised_label_available_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame, {
        "month": f"{month:%Y-%m}",
        "candidate_rows": int(len(frame)),
        "router_top50_identity_exact": True,
        "feature_columns": int(len(fields)),
        "feature_complete_rows": int(frame.loc[:, list(fields)].notna().all(axis=1).sum()),
        "policy_valid_rows": int(frame["policy_ordinal_valid"].fillna(False).astype(bool).sum()),
        "normalised_valid_rows": int(frame["normalised_magnitude_valid"].fillna(False).astype(bool).sum()),
    }


def _load_window(*, candidate_root: Path | None, feature_root: Path | Sequence[Path], label_root: Path, router_root: Path,
                 start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in _months_between(start, end):
        frame, audit = _join_month(
            candidate_root=candidate_root, feature_root=feature_root, label_root=label_root,
            router_root=router_root, month=month, fields=fields,
        )
        pieces.append(frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy())
        audits.append(audit)
    result = pd.concat(pieces, ignore_index=True)
    if result.empty or result["candidate_id"].duplicated().any():
        raise AssertionError("empty or duplicate strict window")
    return result, audits


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    work = scored.merge(
        labels.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    work["policy_net_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    work["policy_ordinal_valid"] = work["policy_ordinal_valid"].fillna(False).astype(bool)
    work["__rank__"] = work.groupby("__decision_ts__", sort=False)["base_score"].rank(method="first", ascending=False)
    work["__count__"] = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    result: dict[str, float] = {}
    top5_ts: pd.Series | None = None
    for fraction, token in ((.01, "1"), (.02, "2"), (.05, "5"), (.10, "10"), (.15, "15"), (.20, "20")):
        selected = work.loc[work["__rank__"].le(np.ceil(work["__count__"] * fraction))].copy()
        valid = selected.loc[selected["policy_ordinal_valid"] & np.isfinite(selected["policy_net_bps"])].copy()
        per_ts = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
        result[f"dtp{token}_bps"] = float(per_ts.mean()) if len(per_ts) else np.nan
        result[f"top{token}_outcome_coverage"] = float(len(valid) / len(selected)) if len(selected) else np.nan
        result[f"top{token}_precision50"] = float((valid["policy_net_bps"] > 50.0).mean()) if len(valid) else np.nan
        result[f"top{token}_precision100"] = float((valid["policy_net_bps"] > 100.0).mean()) if len(valid) else np.nan
        if fraction == .05:
            top5_ts = per_ts
        if fraction in (.05, .10, .20):
            for threshold in (50.0, 100.0):
                winners = work.loc[work["policy_ordinal_valid"] & np.isfinite(work["policy_net_bps"]) & work["policy_net_bps"].gt(threshold)].copy()
                winners["__selected__"] = winners["__rank__"].le(np.ceil(winners["__count__"] * fraction))
                by_ts = winners.groupby("__decision_ts__", sort=False)["__selected__"].mean()
                result[f"recall{int(threshold)}_at{token}"] = float(by_ts.mean()) if len(by_ts) else np.nan
                excess = work.loc[work["policy_ordinal_valid"] & np.isfinite(work["policy_net_bps"])].copy()
                excess["__value__"] = np.maximum(excess["policy_net_bps"].to_numpy(float) - threshold, 0.0)
                numer = excess.loc[excess["__rank__"].le(np.ceil(excess["__count__"] * fraction))].groupby("__decision_ts__", sort=False)["__value__"].sum()
                denom = excess.groupby("__decision_ts__", sort=False)["__value__"].sum()
                economic = (numer / denom).replace([np.inf, -np.inf], np.nan).dropna()
                result[f"er{int(threshold)}_at{token}"] = float(economic.mean()) if len(economic) else np.nan
    for lower, upper in ((0.0, .05), (.05, .10), (.10, .15), (.15, .20)):
        band = work.loc[
            work["__rank__"].gt(np.floor(work["__count__"] * lower))
            & work["__rank__"].le(np.ceil(work["__count__"] * upper))
            & work["policy_ordinal_valid"] & np.isfinite(work["policy_net_bps"]),
        ]
        per_ts = band.groupby("__decision_ts__", sort=False)["policy_net_bps"].mean()
        result[f"band_{int(lower * 100)}_{int(upper * 100)}_bps"] = float(per_ts.mean()) if len(per_ts) else np.nan
    if top5_ts is None or top5_ts.empty:
        result.update({"q10_week_dtp5_bps": np.nan, "q25_month_dtp5_bps": np.nan, "positive_month_fraction_dtp5": np.nan, "worst_fold_dtp5_bps": np.nan})
    else:
        weekly = top5_ts.groupby(top5_ts.index.isocalendar().year.astype(str) + "-" + top5_ts.index.isocalendar().week.astype(str)).mean()
        monthly = top5_ts.groupby(top5_ts.index.tz_localize(None).to_period("M")).mean()
        result["q10_week_dtp5_bps"] = float(weekly.quantile(.10))
        result["q25_month_dtp5_bps"] = float(monthly.quantile(.25))
        result["positive_month_fraction_dtp5"] = float((monthly > 0.0).mean())
        result["worst_fold_dtp5_bps"] = float(monthly.min())
    return result


def _screen_summary(folds: pd.DataFrame) -> pd.DataFrame:
    aggregate = folds.groupby("target", sort=True).mean(numeric_only=True).reset_index()
    control = aggregate.loc[aggregate["target"].eq("t0_policy_ordinal")]
    if len(control) != 1:
        raise AssertionError("missing T0 policy-ordinal control")
    c = control.iloc[0]
    tip = .25 * aggregate["dtp1_bps"] + .25 * aggregate["dtp2_bps"] + .50 * aggregate["dtp5_bps"]
    breadth = (
        .30 * aggregate["er50_at20"] + .25 * aggregate["recall50_at20"]
        + .25 * aggregate["recall100_at20"] + .20 * aggregate["er100_at20"]
    )
    stability = .5 * aggregate["q10_week_dtp5_bps"] + .5 * aggregate["q25_month_dtp5_bps"]
    control_tip = float(.25 * c.dtp1_bps + .25 * c.dtp2_bps + .50 * c.dtp5_bps)
    control_breadth = float(.30 * c.er50_at20 + .25 * c.recall50_at20 + .25 * c.recall100_at20 + .20 * c.er100_at20)
    control_stability = float(.5 * c.q10_week_dtp5_bps + .5 * c.q25_month_dtp5_bps)
    aggregate["tip_bps"] = tip
    aggregate["breadth"] = breadth
    aggregate["stability_bps"] = stability
    aggregate["tip_vs_t0"] = tip / control_tip if control_tip > 0 else np.nan
    aggregate["breadth_vs_t0"] = breadth / control_breadth if control_breadth > 0 else np.nan
    aggregate["stability_vs_t0"] = stability / control_stability if control_stability > 0 else np.nan
    aggregate["base_screen"] = .50 * aggregate["tip_vs_t0"] + .35 * aggregate["breadth_vs_t0"] + .15 * aggregate["stability_vs_t0"]
    aggregate["passes_tip_gate"] = (
        aggregate["dtp1_bps"].ge(.97 * c.dtp1_bps)
        & aggregate["dtp2_bps"].ge(.98 * c.dtp2_bps)
        & aggregate["dtp5_bps"].ge(.98 * c.dtp5_bps)
    )
    aggregate["passes_stability_gate"] = (
        aggregate["q10_week_dtp5_bps"].ge(c.q10_week_dtp5_bps - 1e-9)
        & aggregate["q25_month_dtp5_bps"].ge(c.q25_month_dtp5_bps - 1e-9)
    )
    aggregate["advances"] = aggregate["passes_tip_gate"] & aggregate["passes_stability_gate"]
    return aggregate.sort_values(["advances", "base_screen", "target"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def run(*, candidate_root: Path | None, feature_root: Path | Sequence[Path], label_root: Path, router_root: Path,
        selection_receipt: Path, out: Path, held_months: Sequence[pd.Timestamp],
        train_months: int, reserve_days: int, train_cap: int, n_jobs: int,
        targets: Sequence[str], gain_name: str, truncation: int, sigmoid: float,
        objective: str, hpo_config: Path | None = None) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if not held_months or tuple(sorted(held_months)) != tuple(held_months):
        raise ValueError("held months must be sorted and non-empty")
    fields = _load_f72_fields(selection_receipt)
    unknown = sorted(set(targets).difference(TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    if "t0_policy_ordinal" not in targets:
        raise ValueError("Stage-A screen requires t0_policy_ordinal as its fixed single-base control")
    if gain_name not in GAIN_SCHEDULES:
        raise ValueError(f"unknown gain schedule: {gain_name}")
    if truncation < 1:
        raise ValueError("truncation must be positive")
    if not np.isfinite(sigmoid) or sigmoid <= 0.0:
        raise ValueError("sigmoid must be positive and finite")
    if objective not in {"lambdarank", "rank_xendcg"}:
        raise ValueError(f"unsupported rank objective: {objective}")
    label_gain = GAIN_SCHEDULES[gain_name]
    frozen_hpo, frozen_hpo_receipt = _load_frozen_hpo_config(hpo_config)
    if frozen_hpo_receipt is not None:
        locked = frozen_hpo_receipt["contract"]
        expected = {
            "target": None,
            "objective": objective,
            "gain_name": gain_name,
            "truncation": truncation,
            "sigmoid": sigmoid,
            "fields": list(fields),
        }
        for key, value in locked.items():
            if key == "target":
                if value not in targets:
                    raise AssertionError("frozen HPO target is absent from the requested Base run")
                continue
            if key == "fields":
                if value != expected[key]:
                    raise AssertionError("frozen HPO feature contract differs from Base selection receipt")
            elif value is not None and value != expected[key]:
                raise AssertionError(f"frozen HPO {key} differs from immutable Base contract")
    out.mkdir(parents=True)
    all_metrics: list[dict[str, object]] = []
    all_coverage: list[dict[str, object]] = []
    fold_audits: list[dict[str, object]] = []
    source_hashes: list[Path] = []

    for fold_index, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=reserve_days)
        train_start = reserve - pd.DateOffset(months=train_months)
        held_end = _month_end(held_month)
        window, coverage = _load_window(
            candidate_root=candidate_root, feature_root=feature_root, label_root=label_root,
            router_root=router_root, start=train_start, end=held_end, fields=fields,
        )
        all_coverage.extend(coverage)
        for month in _months_between(train_start, held_end):
            if candidate_root is not None:
                source_hashes.append(_candidate_path(candidate_root, month))
            source_hashes.extend((_feature_path(feature_root, month), _label_path(label_root, month), _router_path(router_root, month)))
        held = window.loc[window["__decision_ts__"].ge(held_month) & window["__decision_ts__"].lt(held_end)].copy()
        if held.empty:
            raise AssertionError(f"{held_month:%Y-%m}: no held routed candidates")
        for target_name in targets:
            spec = TARGETS[target_name]
            valid = window[spec.valid_column].fillna(False).astype(bool)
            available = pd.to_datetime(window[spec.available_column], utc=True, errors="coerce")
            numeric = np.isfinite(pd.to_numeric(window[spec.value_column], errors="coerce"))
            train = window.loc[
                window["__decision_ts__"].lt(reserve) & valid & numeric & available.lt(reserve),
            ].copy()
            train = _sample_complete_queries(train, train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
            if len(train) < 8_000 or train["__decision_ts__"].nunique() < 40:
                raise AssertionError(f"{target_name}/{held_month:%Y-%m}: insufficient strict routed train support")
            labels_train, label_description = _target_labels(train, held, spec)
            x_train, medians = _numeric_matrix(train, fields)
            x_held, _ = _numeric_matrix(held, fields, medians)
            model_params: dict[str, object] = dict(
                objective=objective, metric="ndcg", n_estimators=140, learning_rate=.05,
                max_depth=4, num_leaves=15, min_child_samples=260,
                subsample=.80, subsample_freq=1, colsample_bytree=.80,
                reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001,
                random_state=SEED + fold_index, n_jobs=n_jobs,
                deterministic=True, force_col_wise=True, verbosity=-1,
            )
            if frozen_hpo and target_name == frozen_hpo_receipt["contract"]["target"]:
                model_params.update(
                    learning_rate=frozen_hpo["learning_rate"],
                    max_depth=frozen_hpo["max_depth"],
                    num_leaves=frozen_hpo["num_leaves"],
                    min_child_samples=max(40, int(round(len(train) * frozen_hpo["min_data_fraction"]))),
                    subsample=frozen_hpo["bagging_fraction"],
                    colsample_bytree=frozen_hpo["feature_fraction"],
                    reg_alpha=frozen_hpo["lambda_l1"],
                    reg_lambda=frozen_hpo["lambda_l2"],
                    min_split_gain=frozen_hpo["min_gain_to_split"],
                )
            if objective == "lambdarank":
                model_params.update(
                    lambdarank_truncation_level=truncation,
                    label_gain=label_gain,
                    sigmoid=sigmoid,
                )
            model = LGBMRanker(**model_params)
            model.fit(x_train, labels_train, group=_query_groups(train))
            target_out = out / "target_free_scores" / target_name
            target_out.mkdir(parents=True, exist_ok=True)
            target_free = held.loc[:, list(IDENTITY)].copy()
            target_free["base_score"] = model.predict(x_held).astype(np.float32)
            target_free["base_rank_ts"] = _rank_desc(target_free, "base_score")
            forbidden = [column for column in target_free if any(token in column.lower() for token in PROHIBITED_SCORE_TOKENS)]
            if forbidden:
                raise AssertionError(f"target-free score carries prohibited columns {forbidden}")
            score_path = target_out / f"month={held_month:%Y-%m}.parquet"
            target_free.to_parquet(score_path, index=False, compression="zstd")
            # Outcome diagnostics begin only after the immutable held score is
            # written.  Ranking is over *all* target-free routed candidates;
            # invalid outcomes do not disappear before the score is ranked.
            label_columns = ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]
            metrics = _metrics(target_free, held.loc[:, label_columns])
            all_metrics.append({
                "target": target_name,
                "held_month": f"{held_month:%Y-%m}",
                "train_start": train_start,
                "reserve_start": reserve,
                "train_rows": int(len(train)),
                "train_queries": int(train["__decision_ts__"].nunique()),
                "held_rows": int(len(held)),
                "held_queries": int(held["__decision_ts__"].nunique()),
                "route_fraction": 1.0,
                "target_valid_train_rows": int(valid.loc[train.index].sum()),
                "label_description": json.dumps(label_description, sort_keys=True),
                **metrics,
            })
            fold_audits.append({
                "target": target_name,
                "held_month": f"{held_month:%Y-%m}",
                "router_top50_identity_exact": True,
                "base_receives_router_numeric_input": False,
                "base_post_router_cutoff": False,
                "train_rows_router_selected": int(len(train)),
                "train_label_available_before_reserve": bool(available.loc[train.index].lt(reserve).all()),
                "held_score_target_free": True,
                "held_score_path": str(score_path),
                "feature_count": len(fields),
                "feature_medians_fit_on_train_only": True,
                "label_description": label_description,
                "model_params": {
                    key: model_params[key] for key in (
                        "n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain",
                    )
                },
            })
            _progress(out, event="fold_complete", target=target_name, month=f"{held_month:%Y-%m}", rows=len(held))
            del model, x_train, x_held, train, labels_train, target_free
            gc.collect()
        del window, held
        gc.collect()

    fold_metrics = pd.DataFrame(all_metrics)
    fold_metrics.to_parquet(out / "fold_metrics.parquet", index=False, compression="zstd")
    screen = _screen_summary(fold_metrics)
    screen.to_parquet(out / "screen_summary.parquet", index=False, compression="zstd")
    advancing = screen.loc[screen["advances"], "target"].head(3).tolist()
    if "t0_policy_ordinal" not in advancing:
        advancing = ["t0_policy_ordinal", *advancing][:3]
    pd.DataFrame(all_coverage).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "selection.json", {
        "stage": "A_target_geometry_or_loss_geometry",
        "control": "t0_policy_ordinal",
        "advancing_targets": advancing,
        "selection_rule": "BASE_SCREEN after predeclared DTP1/2/5 and weekly/monthly DTP5 gates",
        "screen_metrics": "timestamp-local and equal-timestamp weighted; invalid held outcomes remain selected but excluded from realised-outcome aggregation with coverage reported",
    })
    _exclusive_json(out / "correctness_report.json", {
        "all_router_top50_identity_exact": bool(all(item["router_top50_identity_exact"] for item in fold_audits)),
        "any_router_numeric_base_input": False,
        "any_post_router_base_cutoff": False,
        "all_train_labels_resolved_before_reserve": bool(all(item["train_label_available_before_reserve"] for item in fold_audits)),
        "all_held_scores_target_free": bool(all(item["held_score_target_free"] for item in fold_audits)),
        "all_feature_medians_train_only": bool(all(item["feature_medians_fit_on_train_only"] for item in fold_audits)),
        "costs_reused_from_canonical_policy_labels_only": True,
        "scope": "offline prescreen only; R/U, MC1, admission, portfolio, live and exchange contracts unchanged",
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-geometry Stage A; no R/U, MC1, portfolio, live or exchange mutation",
        "architecture": "Router top50 gate -> one Base scores all routed rows; no router numeric Base input; no Base post-route cutoff",
        "targets": {key: TARGETS[key].__dict__ for key in targets},
        "features": list(fields),
        "feature_selection_receipt": str(selection_receipt),
        "feature_selection_sha256": _sha256([selection_receipt]),
        "model": {
            "objective": objective, "gain_name": gain_name,
            "label_gain": label_gain if objective == "lambdarank" else "not_used_by_rank_xendcg",
            "base_defaults": {"n_estimators": 140, "learning_rate": .05, "max_depth": 4, "num_leaves": 15, "min_child_samples": 260},
            "frozen_hpo": frozen_hpo or None,
            "frozen_hpo_receipt": frozen_hpo_receipt,
            "frozen_hpo_applies_only_to_target": (
                frozen_hpo_receipt["contract"].get("target") if frozen_hpo_receipt else None
            ),
            "truncation": truncation if objective == "lambdarank" else "not_used_by_rank_xendcg",
            "sigmoid": sigmoid if objective == "lambdarank" else "not_used_by_rank_xendcg",
        },
        "query": "decision timestamp x long side",
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "held_months": [f"{item:%Y-%m}" for item in held_months],
        "candidate_root": str(candidate_root) if candidate_root is not None else None,
        "candidate_population": (
            "direct exact frozen Router50 identities" if candidate_root is None
            else "immutable supplied target-free candidate identities"
        ),
        "feature_roots": [str(item) for item in ((feature_root,) if isinstance(feature_root, Path) else feature_root)],
        "label_root": str(label_root), "router_root": str(router_root),
        "input_sha256": _sha256(source_hashes),
        "fold_audits": fold_audits,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--router-derived-candidates", action="store_true")
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--feature-roots", default=None, help="comma-separated immutable feature roots; exactly one must own each month")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, default=F72_SELECTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=",".join(DEFAULT_HELD_MONTHS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--gain-name", choices=tuple(GAIN_SCHEDULES), default=DEFAULT_GAIN_NAME)
    parser.add_argument("--truncation", type=int, default=12)
    parser.add_argument("--sigmoid", type=float, default=1.0)
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), default="lambdarank")
    parser.add_argument(
        "--hpo-config", type=Path, default=None,
        help="sealed local-HPO run_manifest.json; may alter only the named target's tree/regularisation fields",
    )
    args = parser.parse_args()
    if (args.candidate_root is None) == (not args.router_derived_candidates):
        raise ValueError("supply --candidate-root or --router-derived-candidates, but not both")
    if bool(args.feature_root) == bool(args.feature_roots):
        raise ValueError("supply exactly one of --feature-root or --feature-roots")
    held_months = tuple(_utc(f"{token.strip()}-01") for token in args.held_months.split(",") if token.strip())
    targets = tuple(token.strip() for token in args.targets.split(",") if token.strip())
    print(run(
        candidate_root=args.candidate_root.resolve() if args.candidate_root else None,
        feature_root=(
            args.feature_root.resolve() if args.feature_root else
            tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip())
        ),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(),
        selection_receipt=args.selection_receipt.resolve(), out=args.out.resolve(),
        held_months=held_months, train_months=args.train_months,
        reserve_days=args.reserve_days, train_cap=args.train_cap, n_jobs=args.n_jobs, targets=targets,
        gain_name=args.gain_name, truncation=args.truncation, sigmoid=args.sigmoid,
        objective=args.objective, hpo_config=args.hpo_config.resolve() if args.hpo_config else None,
    ))


if __name__ == "__main__":
    main()
