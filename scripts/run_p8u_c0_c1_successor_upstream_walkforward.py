#!/usr/bin/env python3
"""Build the strict-prequential Router50 -> F72 Base -> Under-F120 successor.

This is a *separately named* September 2026 source-aligned rebuild.  It never
loads a discarded historical booster or score ledger.  The only inputs are:

* full-universe point-in-time feature vectors generated at ``__ts__``;
* frozen, hash-bound Router30 / F72 / Under-F120 field contracts; and
* exact one-minute rich-policy labels joined only after target-free identities
  are fixed.

The early OOF months intentionally have limited support because the repaired
exact label ledger begins in May.  That limitation is written into every fold
receipt; it is not hidden by borrowing legacy labels.  The final package has
no MC1, admission, portfolio, execution, or exchange authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker, early_stopping
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_model_package import (  # noqa: E402
    BASE_GEOMETRY,
    MODEL_ROLES,
    P8UModelBundle,
    SCHEMA,
    _ModelState,
    add_base_geometry,
    role_file_entry,
    sha256_file,
    timestamp_desc_rank,
)
from extreme_price_movements.inference.p8u_production_contract import (  # noqa: E402
    IDENTITY_COLUMNS,
    exact_timestamp_route,
    routed_only,
)


SEED = 1729
IDENTITY = tuple(IDENTITY_COLUMNS)
RESERVE_DAYS = 28
H12_RESOLUTION = pd.Timedelta(hours=12, minutes=5)
ROUTER50 = 0.50
FEATURE_PLAN = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_feature_plan_20260901_v1/required_feature_plan.json"
FEATURE_ROOT = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_feature_panel_febsep2026_v1"
LABEL_FEBAPR = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_exact1m_labels_febapr2026_v1_pathmfe_sampled_oracle/exact_1m_policy_outcomes.parquet"
LABEL_MAYJUL = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_exact1m_labels_mayjul2026_v2_pathmfe_sampled_oracle/exact_1m_policy_outcomes.parquet"
LABEL_AUG = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_exact1m_labels_aug2026_v4_pathmfe_sampled_oracle/exact_1m_policy_outcomes.parquet"
LABEL_JAN = ROOT / "data_perp/artifacts/p8u_c0_c1_successor_exact1m_labels_jan2026_v1_pathmfe_sampled_oracle/exact_1m_policy_outcomes.parquet"


def _utc(value: object) -> pd.Timestamp:
    out = pd.Timestamp(value)
    return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")


def _sha(path: Path) -> str:
    return sha256_file(path)


def _hash_identity(frame: pd.DataFrame) -> str:
    work = frame.loc[:, list(IDENTITY)].copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    work = work.sort_values(["__decision_ts__", "candidate_id", "side_name"], kind="stable")
    digest = hashlib.sha256()
    for row in work.itertuples(index=False, name=None):
        digest.update("|".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    work = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(work.dtype).encode())
    digest.update(str(tuple(work.shape)).encode())
    digest.update(work.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _mkdir_exclusive(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _read_plan(path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text())
    result = {key: tuple(map(str, raw[key])) for key in ("router_features", "f72_features", "under_features", "full_union")}
    if [len(result[key]) for key in ("router_features", "f72_features", "under_features", "full_union")] != [30, 72, 120, 185]:
        raise AssertionError("frozen feature plan counts changed")
    if any(len(set(values)) != len(values) for values in result.values()):
        raise AssertionError("frozen feature plan contains duplicates")
    return result


def _parts(roots: Sequence[Path]) -> list[Path]:
    """Return immutable parts from one or more non-overlapping feature panels.

    A separately sealed January warm-up panel is intentionally allowed beside
    the February--September successor panel.  We never rewrite or merge the
    panels: duplicate decision identities are rejected in ``_read_features``.
    """
    result: list[Path] = []
    for root in roots:
        parts = sorted((root / "features").glob("part_*.parquet"))
        if not parts:
            raise AssertionError(f"feature panel has no immutable parts: {root}")
        manifest = root / "run_manifest.json"
        if not manifest.is_file():
            raise AssertionError(f"feature panel has no immutable manifest: {root}")
        result.extend(parts)
    return result


def _read_features(roots: Sequence[Path], fields: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [*IDENTITY, "__ts__", "__symbol__", *fields]
    pieces: list[pd.DataFrame] = []
    for path in _parts(roots):
        # Avoid loading the dense 185-column history into every fold.  The
        # range filter after the immutable part read is source-only and cannot
        # affect feature values.
        part = pd.read_parquet(path, columns=columns)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part = part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        raise AssertionError(f"no target-free features in {start}..{end}")
    out = pd.concat(pieces, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if out.duplicated(IDENTITY).any() or not out["side_name"].astype(str).eq("long").all():
        raise AssertionError("feature identities are not unique long-only point-in-time rows")
    if not out["__decision_ts__"].eq(pd.to_datetime(out["__ts__"], utc=True) + pd.Timedelta(hours=1)).all():
        raise AssertionError("feature decision timestamp is not exactly source timestamp + one hour")
    return out


def _read_labels(paths: Sequence[Path]) -> pd.DataFrame:
    wanted = [
        "candidate_id", "decision_timestamp", "net_bps", "outcome_available", "outcome_invalid_reason",
        "path_reached_trailing_activation_0p5atr", "path_mfe_atr_h12",
    ]
    pieces: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"path-complete exact label ledger is required: {path}")
        part = pd.read_parquet(path, columns=wanted)
        part["candidate_id"] = part["candidate_id"].astype(str)
        part["decision_timestamp"] = pd.to_datetime(part["decision_timestamp"], utc=True, errors="raise")
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True)
    if out["candidate_id"].duplicated().any():
        raise AssertionError("exact labels overlap across source ledgers")
    if not out["outcome_available"].fillna(False).astype(bool).eq(np.isfinite(pd.to_numeric(out["net_bps"], errors="coerce"))).all():
        raise AssertionError("exact policy label availability / net mismatch")
    return out.sort_values(["decision_timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _join_labels(target_free: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    # Candidate panel is first fully read and identity-validated.  This is the
    # only label join, and it is deliberately after all target-free source
    # validation above.
    joined = target_free.merge(labels, left_on="candidate_id", right_on="candidate_id", how="left", validate="one_to_one")
    decision = pd.to_datetime(joined["__decision_ts__"], utc=True)
    labelled_ts = pd.to_datetime(joined["decision_timestamp"], utc=True, errors="coerce")
    matched = labelled_ts.notna()
    if matched.any() and not decision.loc[matched].eq(labelled_ts.loc[matched]).all():
        raise AssertionError("exact label decision timestamp differs from target-free feature identity")
    joined["label_available_ts"] = decision + H12_RESOLUTION
    joined["label_valid"] = joined["outcome_available"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["net_bps"], errors="coerce"))
    return joined


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise KeyError(f"missing frozen feature fields: {missing[:8]}")
    value = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if medians is None:
        medians = np.nanmedian(value, axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
    else:
        medians = np.asarray(medians, dtype=np.float32)
    bad = ~np.isfinite(value)
    if bad.any():
        value[bad] = np.broadcast_to(medians, value.shape)[bad]
    return value.astype(np.float32, copy=False), medians


def _qid(frame: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(frame["__decision_ts__"], sort=True)
    if (codes < 0).any():
        raise AssertionError("invalid timestamp query")
    return codes.astype(np.int64)


def _groups(frame: pd.DataFrame) -> list[int]:
    counts = frame.groupby("__decision_ts__", sort=True).size().to_numpy(np.int32)
    if len(counts) < 2 or int(counts.min()) < 2:
        raise AssertionError("ranker requires at least two complete rows per timestamp query")
    return counts.astype(int).tolist()


def _chronological_inner(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stamps = np.array(sorted(frame["__decision_ts__"].unique()))
    split = max(1, min(len(stamps) - 1, int(np.floor(.80 * len(stamps)))))
    fit_stamp = set(stamps[:split])
    fit = frame["__decision_ts__"].isin(fit_stamp).to_numpy(bool)
    valid = ~fit
    if not fit.any() or not valid.any():
        raise AssertionError("insufficient whole-timestamp inner fit/validation split")
    return fit, valid


def _base_grade(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> tuple[np.ndarray, dict[str, float]]:
    if lower is None or upper is None:
        lower, upper = np.nanquantile(values, [.02, .98])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise AssertionError("base target P2/P98 geometry is degenerate")
    edges = np.linspace(float(lower), float(upper), 7)[1:-1]
    return np.digitize(np.clip(values, lower, upper), edges, right=False).astype(np.int32), {"p2": float(lower), "p98": float(upper), "edges": [float(x) for x in edges]}


def _router_grade(values: np.ndarray) -> np.ndarray:
    # Retained P8u floor100/cap250 semantic: non-opportunity <= 100 bps is
    # grade 0; five ordered excess bands handle only tail routing value.
    excess = np.clip(np.asarray(values, dtype=float) - 100.0, 0.0, 250.0)
    out = np.zeros(len(excess), dtype=np.int32)
    positive = excess > 0.0
    out[positive] = np.digitize(excess[positive], [31.25, 62.5, 109.375, 171.875], right=False) + 1
    return out


def _router_weights(labels: np.ndarray) -> np.ndarray:
    # Query-normalise after a smooth tail emphasis, matching the retained
    # Router requirement that every timestamp contributes one total loss mass.
    return 1.0 + np.sqrt(np.asarray(labels, dtype=float)) / max(np.sqrt(5.0), 1e-12)


def _base_weights(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    raw = 1.0 + .125 * np.asarray(labels, dtype=float)
    work = pd.DataFrame({"ts": frame["__decision_ts__"].to_numpy(), "w": raw})
    normalised = work["w"] / work.groupby("ts", sort=False)["w"].transform("mean")
    return np.clip(normalised.to_numpy(float), .5, 2.0).astype(np.float32)


@dataclass
class Fitted:
    model: Any
    fields: tuple[str, ...]
    medians: np.ndarray
    reference: np.ndarray | None
    model_format: str
    audit: dict[str, Any]


def _fit_router(train: pd.DataFrame, fields: tuple[str, ...], *, seed: int) -> Fitted:
    y = _router_grade(train["net_bps"].to_numpy(float))
    x, medians = _matrix(train, fields)
    fit, valid = _chronological_inner(train)
    min_child = max(500, int(np.ceil(.017038 * int(fit.sum()))))
    model = LGBMRanker(
        objective="rank_xendcg", metric="ndcg", n_estimators=1000, learning_rate=.0567571,
        max_depth=4, num_leaves=15, min_child_samples=min_child, min_split_gain=.00321538,
        colsample_bytree=.787355, subsample=.727909, reg_alpha=.0141675, reg_lambda=.216746,
        max_bin=127, label_gain=[0, 1, 2, 4, 7, 11], lambdarank_truncation_level=12,
        random_state=seed, n_jobs=min(4, os.cpu_count() or 1), verbosity=-1,
    )
    model.fit(
        x[fit], y[fit], group=_groups(train.loc[fit]), sample_weight=_router_weights(y[fit]),
        eval_set=[(x[valid], y[valid])], eval_group=[_groups(train.loc[valid])],
        callbacks=[early_stopping(30, verbose=False)],
    )
    raw_ref = model.predict(x[fit]).astype(np.float32)
    return Fitted(model, fields, medians, np.sort(raw_ref), "lightgbm_booster_text", {
        "rows": int(len(train)), "fit_rows": int(fit.sum()), "valid_rows": int(valid.sum()),
        "queries": int(train["__decision_ts__"].nunique()), "classes": int(np.max(y) + 1),
        "label_sha256": _hash_array(y), "feature_sha256": _hash_array(x), "identity_sha256": _hash_identity(train),
        "min_child_samples": int(min_child), "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "target": "P8u_floor100_cap250", "objective": "rank_xendcg",
    })


def _fit_base(train: pd.DataFrame, fields: tuple[str, ...], *, seed: int) -> Fitted:
    y, geometry = _base_grade(train["net_bps"].to_numpy(float))
    x, medians = _matrix(train, fields)
    fit, valid = _chronological_inner(train)
    weights = _base_weights(train, y)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=.06509939203594448, depth=5, l2_leaf_reg=2.2357264515179467,
        random_strength=.9428901899180999, rsm=.80065065628642,
        bootstrap_type="Bernoulli", subsample=.7096047498424234, random_seed=seed,
        thread_count=min(4, os.cpu_count() or 1), verbose=False, allow_writing_files=False,
        od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x[fit], y[fit], group_id=_qid(train.loc[fit]), weight=weights[fit]),
        eval_set=Pool(x[valid], y[valid], group_id=_qid(train.loc[valid]), weight=weights[valid]),
        use_best_model=True, verbose=False,
    )
    return Fitted(model, fields, medians, None, "catboost_ranker_cbm", {
        "rows": int(len(train)), "fit_rows": int(fit.sum()), "valid_rows": int(valid.sum()),
        "queries": int(train["__decision_ts__"].nunique()), "classes": int(np.max(y) + 1),
        "label_sha256": _hash_array(y), "feature_sha256": _hash_array(x), "identity_sha256": _hash_identity(train),
        "weight_sha256": _hash_array(weights), "best_iteration": int(model.tree_count_),
        "target": "raw_exact_policy_net_p2p98_equal_width6", "target_geometry": geometry,
        "objective": "QueryRMSE", "weights": "tail_linear_125_query_normalised_clip_0.5_2",
    })


def _score_router(fitted: Fitted, frame: pd.DataFrame) -> pd.DataFrame:
    x, _ = _matrix(frame, fitted.fields, fitted.medians)
    raw = np.asarray(fitted.model.predict(x), dtype=np.float32)
    ref = fitted.reference
    if ref is None:
        raise AssertionError("router has no training rank reference")
    left, right = np.searchsorted(ref, raw, "left"), np.searchsorted(ref, raw, "right")
    rank = np.clip(((left + right) * .5 + .5) / len(ref), 0., 1.).astype(np.float32)
    out = frame.loc[:, list(IDENTITY)].copy()
    out["router_raw_score"] = raw
    out["router_primary_rank"] = rank
    out["router_score"] = rank
    return out


def _router50(frame: pd.DataFrame, router_score: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(router_score, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined["router_score"].isna().any() or len(joined) != len(frame):
        raise AssertionError("router score does not cover full point-in-time universe")
    routed = exact_timestamp_route(joined, score_column="router_score", fraction=ROUTER50)
    eligible = routed["router50_eligible"].fillna(False).astype(bool)
    if not eligible.any() or not np.allclose(routed.loc[eligible, "router_fraction"], ROUTER50, rtol=0.0, atol=0.0):
        raise AssertionError("Router50 carrier does not declare the frozen eligible identities")
    return routed


def _score_base(fitted: Fitted, routed: pd.DataFrame) -> pd.DataFrame:
    work = routed_only(routed)
    x, _ = _matrix(work, fitted.fields, fitted.medians)
    out = work.loc[:, list(IDENTITY)].copy()
    out["base_score"] = np.asarray(fitted.model.predict(x), dtype=np.float32)
    out["base_rank_ts"] = timestamp_desc_rank(out, "base_score")
    return out


def _anchor_prequential(frame: pd.DataFrame) -> np.ndarray:
    """14-day blockwise prequential Base-rank -> exact-EV anchor."""
    work = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    stamps = pd.to_datetime(work["__decision_ts__"], utc=True)
    block = ((stamps - stamps.min()).dt.total_seconds() // (14 * 86400)).astype(int).to_numpy()
    result = np.full(len(work), np.nan, dtype=np.float32)
    for key in np.unique(block):
        here = np.flatnonzero(block == key)
        prior = np.flatnonzero(block < key)
        prior = prior[work.loc[prior, "label_valid"].to_numpy(bool)]
        if len(prior) < 500:
            continue
        x = pd.to_numeric(work.loc[prior, "base_rank_ts"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(work.loc[prior, "net_bps"], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 500 or np.unique(x[valid]).size < 5:
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x[valid], y[valid])
        current = pd.to_numeric(work.loc[here, "base_rank_ts"], errors="coerce").to_numpy(float)
        current_valid = np.isfinite(current)
        result[here[current_valid]] = iso.predict(current[current_valid]).astype(np.float32)
    return pd.Series(result, index=work.index).reindex(frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").index).to_numpy(np.float32)


UNDER_TARGETS = {
    "residual_path100": "prequential Base-rank residual >= +100 bps with observed H12 MFE >= 0.5 ATR",
    "residual100": "prequential Base-rank residual >= +100 bps",
    "policy50": "exact rich-policy net >= +50 bps",
    "policy100": "exact rich-policy net >= +100 bps",
}


def _fit_under(
    train: pd.DataFrame, fields: tuple[str, ...], *, seed: int, target_mode: str,
) -> Fitted:
    if target_mode not in UNDER_TARGETS:
        raise ValueError(f"unknown Under target mode: {target_mode}")
    work = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True).copy()
    # Base-query geometry is defined over the *complete Router50 timestamp*,
    # before any outcome/path target filter.  Rebuilding it after filtering
    # would change rank meaning and is explicitly rejected by the inference
    # contract.
    scored_all = add_base_geometry(work)
    anchor = _anchor_prequential(work)
    path_ok = work["path_reached_trailing_activation_0p5atr"].fillna(False).astype(bool).to_numpy()
    valid_label = work["label_valid"].to_numpy(bool)
    if target_mode == "residual_path100":
        valid = valid_label & np.isfinite(anchor) & path_ok
    elif target_mode == "residual100":
        valid = valid_label & np.isfinite(anchor)
    else:
        valid = valid_label
    work = scored_all.loc[valid].reset_index(drop=True)
    anchor = anchor[valid]
    net = work["net_bps"].to_numpy(float)
    if target_mode.startswith("residual"):
        y = (net - anchor >= 100.).astype(np.int32)
    elif target_mode == "policy50":
        y = (net >= 50.).astype(np.int32)
    else:
        y = (net >= 100.).astype(np.int32)
    if len(work) < 2_000 or y.min() == y.max():
        raise AssertionError("Under lacks sufficient strict-prequential support after path activation")
    full_fields = tuple((*BASE_GEOMETRY, *fields))
    x, medians = _matrix(work, full_fields)
    model = LGBMRanker(
        objective="rank_xendcg", metric="ndcg", n_estimators=260, learning_rate=.045,
        max_depth=4, num_leaves=15, min_child_samples=350, min_split_gain=.001,
        colsample_bytree=.80, subsample=.82, reg_alpha=.02, reg_lambda=8., max_bin=255,
        label_gain=[0, 1, 2, 4, 7, 11, 16, 24], lambdarank_truncation_level=12,
        random_state=seed, n_jobs=min(4, os.cpu_count() or 1), verbosity=-1,
    )
    model.fit(x, y, group=_groups(work))
    return Fitted(model, full_fields, medians, None, "lightgbm_booster_text", {
        "rows_before_prequential_anchor": int(len(train)), "rows": int(len(work)),
        "queries": int(work["__decision_ts__"].nunique()), "positive_fraction": float(y.mean()),
        "label_sha256": _hash_array(y), "feature_sha256": _hash_array(x), "identity_sha256": _hash_identity(work),
        "target": target_mode, "target_definition": UNDER_TARGETS[target_mode],
        "anchor": (
            "14d blockwise prequential isotonic Base-rank-to-exact-policy-net"
            if target_mode.startswith("residual") else "not_used_direct_policy_target"
        ),
        "path_requirement": "full observed H12 MFE >= 0.5 ATR" if target_mode == "residual_path100" else "not_used",
        "objective": "rank_xendcg",
    })


def _score_under(fitted: Fitted, routed: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    work = routed_only(routed).merge(base, on=list(IDENTITY), how="left", validate="one_to_one")
    if work[["base_score", "base_rank_ts"]].isna().any().any():
        raise AssertionError("Under requires one Base score per Router50 candidate")
    scored = add_base_geometry(work)
    x, _ = _matrix(scored, fitted.fields, fitted.medians)
    out = scored.loc[:, list(IDENTITY)].copy()
    out["under_raw_score"] = np.asarray(fitted.model.predict(x), dtype=np.float32)
    out["under_rank_ts"] = timestamp_desc_rank(out, "under_raw_score")
    return out


def _train_window(labelled: pd.DataFrame, held_start: pd.Timestamp) -> pd.DataFrame:
    cutoff = held_start - pd.Timedelta(days=RESERVE_DAYS)
    result = labelled.loc[
        labelled["__decision_ts__"].lt(cutoff)
        & labelled["label_available_ts"].lt(held_start)
        & labelled["label_valid"]
    ].copy()
    if result.empty:
        raise AssertionError(f"no strict resolved labels before {held_start}")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _metric(score: pd.DataFrame, labels: pd.DataFrame, value: str, *, layer: str, month: str) -> list[dict[str, Any]]:
    work = score.merge(labels.loc[labels["label_valid"], ["candidate_id", "net_bps"]], on="candidate_id", how="inner", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for top in (1, 2, 5, 10):
        ordered = work.sort_values(["__decision_ts__", value, "candidate_id"], ascending=[True, False, True], kind="stable")
        take = ordered.groupby("__decision_ts__", sort=False).head(top)
        rows.append({
            "layer": layer, "month": month, "top_per_timestamp": top,
            "rows": int(len(take)), "timestamps": int(take["__decision_ts__"].nunique()),
            "mean_net_bps": float(take["net_bps"].mean()), "hit_rate_gt0": float((take["net_bps"] > 0).mean()),
            "hit_rate_gt50": float((take["net_bps"] > 50).mean()),
        })
    return rows


def _save_fitted(package: Path, role: str, fitted: Fitted, output_fields: Sequence[str]) -> tuple[_ModelState, dict[str, Any], Any]:
    role_root = package / "models" / role
    role_root.mkdir(parents=True)
    model_path = role_root / ("model.cbm" if fitted.model_format == "catboost_ranker_cbm" else "model.txt")
    state_path = role_root / "state.npz"
    if fitted.model_format == "catboost_ranker_cbm":
        fitted.model.save_model(str(model_path))
    else:
        fitted.model.booster_.save_model(str(model_path))
    state: dict[str, np.ndarray] = {"medians": np.asarray(fitted.medians, dtype=np.float32)}
    if fitted.reference is not None:
        state["rank_reference"] = np.asarray(fitted.reference, dtype=np.float32)
    np.savez_compressed(state_path, **state)
    entry = role_file_entry(
        package, role=role, model_path=model_path, state_path=state_path, model_format=fitted.model_format,
        feature_order=fitted.fields, output_fields=output_fields,
    )
    model_state = _ModelState(role, fitted.fields, fitted.medians, fitted.reference, model_path, fitted.model_format)
    return model_state, entry, fitted.model


def _parity(package: Path, raw: P8UModelBundle, sample: pd.DataFrame) -> dict[str, Any]:
    loaded = P8UModelBundle.load(package, verify_hashes=True)
    before = raw.score_stack(sample)
    after = loaded.score_stack(sample)
    detail: dict[str, Any] = {}
    for name, left, right, cols in (
        ("router", before[0], after[0], ("router_raw_score", "router_primary_rank")),
        ("base", before[1], after[1], ("base_score", "base_rank_ts")),
        ("under", before[2], after[2], ("under_raw_score", "under_rank_ts")),
    ):
        merge = left.merge(right, on=list(IDENTITY), suffixes=("_raw", "_loaded"), validate="one_to_one")
        maxima: dict[str, float] = {}
        for col in cols:
            delta = np.abs(merge[f"{col}_raw"].to_numpy(float) - merge[f"{col}_loaded"].to_numpy(float))
            maxima[col] = float(np.nanmax(delta)) if len(delta) else 0.0
            if not np.allclose(merge[f"{col}_raw"], merge[f"{col}_loaded"], rtol=0., atol=1e-6):
                raise AssertionError(f"{name} reload parity fails {col}")
        detail[name] = {"rows": int(len(merge)), "max_abs_delta": maxima}
    return detail


def _score_fold(
    *, labelled: pd.DataFrame, held: pd.DataFrame, held_start: pd.Timestamp, plan: dict[str, tuple[str, ...]],
    previous_router_oof: list[pd.DataFrame], previous_base_oof: list[pd.DataFrame], seed: int, under_target: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Fitted], dict[str, Any]]:
    router_train = _train_window(labelled, held_start)
    router = _fit_router(router_train, plan["router_features"], seed=seed)
    router_score = _score_router(router, held)
    routed = _router50(held, router_score)
    result: dict[str, pd.DataFrame] = {"router": router_score}
    fits: dict[str, Fitted] = {"router": router}
    audit: dict[str, Any] = {"router": router.audit, "held_target_free_identity_sha256": _hash_identity(held)}
    result["router_population"] = routed

    if previous_router_oof:
        history = pd.concat(previous_router_oof, ignore_index=True)
        history = history.merge(labelled.loc[:, ["candidate_id", "label_valid", "label_available_ts", "net_bps"]], on="candidate_id", how="left", validate="one_to_one")
        base_train = history.loc[
            history["router50_eligible"].fillna(False).astype(bool)
            & history["label_valid"].fillna(False).astype(bool)
            & history["label_available_ts"].lt(held_start)
            & history["__decision_ts__"].lt(held_start - pd.Timedelta(days=RESERVE_DAYS))
        ].copy()
        if len(base_train) >= 2_000 and base_train["__decision_ts__"].nunique() >= 10:
            base = _fit_base(base_train, plan["f72_features"], seed=seed + 101)
            base_score = _score_base(base, routed)
            result["base"] = base_score
            result["base_population"] = routed_only(routed).merge(base_score, on=list(IDENTITY), how="left", validate="one_to_one")
            fits["base"] = base
            audit["base"] = base.audit
            if previous_base_oof:
                prior_base = pd.concat(previous_base_oof, ignore_index=True)
                prior_base = prior_base.merge(labelled.loc[:, ["candidate_id", "label_valid", "label_available_ts", "net_bps", "path_reached_trailing_activation_0p5atr"]], on="candidate_id", how="left", validate="one_to_one")
                # Keep the complete scored Router50 query for Base geometry;
                # only _fit_under's target mask removes invalid outcomes.
                # Otherwise the 52 frozen-source exclusions would silently
                # alter `base_rank_ts` semantics.
                under_train = prior_base.loc[
                    prior_base["__decision_ts__"].lt(held_start - pd.Timedelta(days=RESERVE_DAYS))
                ].copy()
                if len(under_train) >= 2_000 and under_train["__decision_ts__"].nunique() >= 10:
                    try:
                        under = _fit_under(
                            under_train, plan["under_features"], seed=seed + 202,
                            target_mode=under_target,
                        )
                    except AssertionError as exc:
                        # The May-start exact ledger cannot provide a prior
                        # 14-day Base-OOF anchor for the earliest Under hold.
                        # Mark that OOF fold unavailable; do not relax the
                        # prequential anchor or substitute an in-sample score.
                        if "Under lacks sufficient strict-prequential support" not in str(exc):
                            raise
                        audit["under_unavailable"] = {
                            "reason": str(exc), "rows_before_anchor": int(len(under_train)),
                            "strict_fail_closed": True,
                        }
                    else:
                        under_score = _score_under(under, routed, base_score)
                        result["under"] = under_score
                        fits["under"] = under
                        audit["under"] = under.audit
    return result, fits, audit


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out).resolve()
    _mkdir_exclusive(output)
    temporary = output.with_name(f".{output.name}.build-{os.getpid()}")
    temporary.mkdir()
    try:
        plan = _read_plan(Path(args.feature_plan))
        # A separately sealed warm-up ledger may extend the prequential
        # history.  Keep it optional so established replay receipts retain
        # their exact input set unless the caller explicitly opts in.
        # The Oct--Jan ledger includes January by construction.  It therefore
        # *replaces* the standalone January receipt instead of being joined
        # alongside it, preventing duplicated candidate identities from
        # silently changing a supervised population.
        if args.labels_octjan is not None:
            label_paths = [Path(args.labels_octjan)]
        else:
            label_paths = [Path(args.labels_jan)]
        label_paths.extend([
            Path(args.labels_febapr),
            Path(args.labels_mayjul),
            Path(args.labels_aug),
        ])
        labels = _read_labels(label_paths)
        start, end = _utc(args.start), _utc(args.end)
        feature_roots = [Path(root).resolve() for root in args.feature_root]
        feature = _read_features(feature_roots, plan["full_union"], start, end)
        labelled = _join_labels(feature, labels)
        feature_receipts = [
            {"path": str(root), "manifest_sha256": _sha(root / "run_manifest.json")}
            for root in feature_roots
        ]
        _write_json(temporary / "source_audit.json", {
            "target_free_feature_panels": feature_receipts,
            "feature_plan": str(Path(args.feature_plan).resolve()), "feature_plan_sha256": _sha(Path(args.feature_plan)),
            "label_ledgers": [
                {"path": str(path.resolve()), "sha256": _sha(path)} for path in label_paths
            ],
            "candidate_rows": int(len(feature)), "candidate_identity_sha256": _hash_identity(feature),
            "valid_exact_label_rows": int(labelled["label_valid"].sum()), "target_or_outcome_feature_input": False,
            "resolution_guard": str(H12_RESOLUTION), "router_fraction": ROUTER50,
        })

        held_months = [_utc(value) for value in args.held_month]
        held_months = sorted(held_months)
        router_oof: list[pd.DataFrame] = []
        base_oof: list[pd.DataFrame] = []
        fold_audits: dict[str, Any] = {}
        all_scores: list[pd.DataFrame] = []
        metrics: list[dict[str, Any]] = []
        for offset, held_start in enumerate(held_months):
            held_end = held_start + pd.offsets.MonthBegin(1)
            held = feature.loc[feature["__decision_ts__"].ge(held_start) & feature["__decision_ts__"].lt(held_end)].copy()
            if held.empty:
                raise AssertionError(f"no target-free held features for {held_start:%Y-%m}")
            score, _fits, audit = _score_fold(
                labelled=labelled, held=held, held_start=held_start, plan=plan,
                previous_router_oof=router_oof, previous_base_oof=base_oof,
                seed=SEED + offset * 1000, under_target=str(args.under_target),
            )
            month = f"{held_start:%Y-%m}"
            router_full = score["router_population"].loc[:, [*IDENTITY, "router_raw_score", "router_primary_rank", "router50_eligible"]].copy()
            router_oof.append(score["router_population"])
            metrics.extend(_metric(score["router"], labelled, "router_primary_rank", layer="router", month=month))
            fold_score = router_full
            if "base" in score:
                base = score["base"].copy()
                base["router50_eligible"] = True
                base_oof.append(score["base_population"])
                fold_score = fold_score.merge(base.loc[:, [*IDENTITY, "base_score", "base_rank_ts"]], on=list(IDENTITY), how="left", validate="one_to_one")
                metrics.extend(_metric(base, labelled, "base_score", layer="base", month=month))
            if "under" in score:
                under = score["under"].copy()
                joint = score["base"].merge(under, on=list(IDENTITY), validate="one_to_one")
                joint["current_score"] = (
                    (1.0 - float(args.under_weight)) * joint["base_rank_ts"]
                    + float(args.under_weight) * joint["under_rank_ts"]
                )
                fold_score = fold_score.merge(joint.loc[:, [*IDENTITY, "under_raw_score", "under_rank_ts", "current_score"]], on=list(IDENTITY), how="left", validate="one_to_one")
                metrics.extend(_metric(joint, labelled, "current_score", layer="current", month=month))
            all_scores.append(fold_score.assign(fold_month=month))
            fold_audits[month] = audit

        # Persist target-free OOF score identities before metrics are joined or
        # summarised.  These values are the sole permissible upstream input to
        # a later C0/C1 training stage.
        score_columns = [*IDENTITY, "fold_month", "router_raw_score", "router_primary_rank", "router50_eligible", "base_score", "base_rank_ts", "under_raw_score", "under_rank_ts", "current_score"]
        combined = pd.concat(all_scores, ignore_index=True, sort=False)
        for column in score_columns:
            if column not in combined:
                combined[column] = np.nan
        combined = combined.loc[:, score_columns].drop_duplicates([*IDENTITY, "fold_month"], keep="last")
        combined.to_parquet(temporary / "oof_target_free_scores.parquet", index=False, compression="zstd")
        pd.DataFrame(metrics).to_parquet(temporary / "oof_outcome_metrics.parquet", index=False, compression="zstd")
        _write_json(temporary / "fold_audits.json", fold_audits)

        # The final September package only consumes strict OOF Router / Base
        # histories.  It has no in-sample Router selection or Base score in a
        # downstream supervised label.
        final_cutoff = _utc(args.final_cutoff)
        final_train = _train_window(labelled, final_cutoff)
        router = _fit_router(final_train, plan["router_features"], seed=SEED + 10_000)
        router_history = pd.concat(router_oof, ignore_index=True).merge(
            labelled.loc[:, ["candidate_id", "label_valid", "label_available_ts", "net_bps"]], on="candidate_id", how="left", validate="one_to_one"
        )
        base_train = router_history.loc[
            router_history["router50_eligible"].fillna(False).astype(bool)
            & router_history["label_valid"].fillna(False).astype(bool)
            & router_history["label_available_ts"].lt(final_cutoff)
            & router_history["__decision_ts__"].lt(final_cutoff - pd.Timedelta(days=RESERVE_DAYS))
        ].copy()
        base = _fit_base(base_train, plan["f72_features"], seed=SEED + 10_101)
        base_history = pd.concat(base_oof, ignore_index=True).merge(
            labelled.loc[:, ["candidate_id", "label_valid", "label_available_ts", "net_bps", "path_reached_trailing_activation_0p5atr"]], on="candidate_id", how="left", validate="one_to_one"
        )
        # As above, retain every point-in-time Base score in a timestamp
        # query.  Outcome-invalid rows cannot become supervised labels, but
        # must remain in the deterministic Base-rank geometry of valid rows.
        under_train = base_history.loc[
            base_history["__decision_ts__"].lt(final_cutoff - pd.Timedelta(days=RESERVE_DAYS))
        ].copy()
        under = _fit_under(
            under_train, plan["under_features"], seed=SEED + 10_202,
            target_mode=str(args.under_target),
        )

        package = temporary / "model_bundle"
        package.mkdir()
        states: dict[str, _ModelState] = {}
        models: dict[str, Any] = {}
        entries: dict[str, dict[str, Any]] = {}
        for role, fitted, output_fields in (
            ("router_model", router, ("router_raw_score", "router_primary_rank")),
            ("base_model", base, ("base_score", "base_rank_ts")),
            ("under_model", under, ("under_raw_score", "under_rank_ts")),
        ):
            state, entry, model = _save_fitted(package, role, fitted, output_fields)
            states[role], entries[role], models[role] = state, entry, model
        manifest = {
            "schema": SCHEMA, "package_kind": "P8U source-aligned September successor Router50/F72/Under-F120",
            "created_at": datetime.now(timezone.utc).isoformat(), "training_cutoff": final_cutoff.isoformat(), "side": "long",
            "routing": {"fraction": ROUTER50, "scope": "complete point-in-time frozen universe", "tie_break": "candidate_id ascending"},
            "models": entries, "inference_flow": ["router_model", "exact timestamp Router50", "base_model", "under_model"],
            "source_receipts": {
                "feature_plan": {"path": str(Path(args.feature_plan).resolve()), "sha256": _sha(Path(args.feature_plan))},
                "feature_panels": feature_receipts,
                "label_ledgers": [
                    {"path": str(path.resolve()), "sha256": _sha(path)} for path in label_paths
                ],
            },
            "training": {"router_model": router.audit, "base_model": base.audit, "under_model": under.audit},
            "strict_prequential": {
                "reserve_days": RESERVE_DAYS, "resolution_guard": str(H12_RESOLUTION),
                "router_to_base": "Base training rows are selected by prior Router OOF identities only.",
                "base_to_under": "Under training rows use prior Base OOF scores only; 14d anchor uses earlier blocks only.",
            },
            "under_target": {
                "name": str(args.under_target),
                "definition": UNDER_TARGETS[str(args.under_target)],
            },
            "under_direct_score_authority_weight": float(args.under_weight),
            "known_limitations": [
                "Exact source-aligned labels begin in January 2026; early OOF support is intentionally shorter and reported per fold.",
                "This package excludes MC1, admission, portfolio, exact execution adjustments, and exchange authority.",
                "It is a separately named rebuild, not bit parity with deleted historical Router/Base/Under models.",
            ],
        }
        _write_json(package / "manifest.json", manifest)
        raw_bundle = P8UModelBundle(root=package, manifest=manifest, states=states, models=models)
        parity_start = final_cutoff - pd.Timedelta(hours=24)
        parity_feature = feature.loc[feature["__decision_ts__"].ge(parity_start) & feature["__decision_ts__"].lt(final_cutoff)].copy()
        parity = _parity(package, raw_bundle, parity_feature)
        _write_json(package / "parity_report.json", {"status": "pass", "target_free_rows": int(len(parity_feature)), "detail": parity})
        _write_json(temporary / "run_manifest.json", {
            "schema": "p8u_c0_c1_successor_upstream_walkforward_v1", "status": "complete",
            "scope": "offline source-aligned Router/Base/Under rebuild; no MC1/admission/portfolio/exchange operation",
            "model_bundle": str(package.relative_to(temporary)), "model_bundle_manifest_sha256": _sha(package / "manifest.json"),
            "oof_score_rows": int(len(combined)), "oof_metric_rows": int(len(metrics)), "final_cutoff": final_cutoff.isoformat(),
            "target_free_scores_persisted_before_metrics": True, "router50_only_downstream": True,
        })
        output.rmdir()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root", type=Path, action="append", default=None,
        help="Immutable target-free feature panel. Repeat only for sealed non-overlapping warm-up panels.",
    )
    parser.add_argument("--feature-plan", type=Path, default=FEATURE_PLAN)
    parser.add_argument("--labels-jan", type=Path, default=LABEL_JAN)
    parser.add_argument(
        "--labels-octjan", type=Path, default=None,
        help="Optional immutable Oct--Jan warm-up label ledger; never substituted for existing ledgers.",
    )
    parser.add_argument("--labels-febapr", type=Path, default=LABEL_FEBAPR)
    parser.add_argument("--labels-mayjul", type=Path, default=LABEL_MAYJUL)
    parser.add_argument("--labels-aug", type=Path, default=LABEL_AUG)
    parser.add_argument("--under-target", choices=tuple(UNDER_TARGETS), default="residual_path100")
    parser.add_argument(
        "--under-weight", type=float, default=0.25,
        help="Direct ranking authority for Under. Use 0 for telemetry-only Under output.",
    )
    parser.add_argument("--start", default="2026-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-09-02T00:00:00Z")
    parser.add_argument("--held-month", action="append", default=None)
    parser.add_argument("--final-cutoff", default="2026-09-02T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= float(args.under_weight) <= 1.0:
        raise SystemExit("--under-weight must be in [0, 1]")
    if not args.feature_root:
        args.feature_root = [FEATURE_ROOT]
    # ``argparse`` appends explicit values to a non-empty default.  Make an
    # explicit fold list authoritative so a caller cannot silently repeat
    # June--August after adding the required May fold.
    if not args.held_month:
        args.held_month = ["2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"]
    print(run(args))


if __name__ == "__main__":
    main()
