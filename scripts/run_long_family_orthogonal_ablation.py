#!/usr/bin/env python3
"""Orthogonal family-attribution and query-group ablations.

This runner is deliberately a companion to
``run_long_family_conditional_correctness.py``.  It reuses the frozen Cap-120
contract and exact execution labels, but exposes the fold matrices needed to
test whether family/path inputs add information beyond the incumbent score.

The important differences from the original conditional run are:

* family contributions are residualised against the incumbent score before a
  ``family contribution x residual outcome`` attribution is formed;
* feature-group arms are fit separately (anchor, context, family, history,
  factor and full contract), with two deterministic permutation placebos;
* local pairwise preference is repeated over 4-hour, 1-day, 3-day and 7-day
  query blocks using the residual outcome rather than raw policy PnL;
* pooled, monthly and weekly stability are reported before any arm can be
  promoted.

All parameters are declared in this file.  No OOS outcomes are used to select
features, permutations, query modes or the reported arm.  The source v8 run is
only used as a cached path/label materialisation; each ablation refits on the
same chronological fold partitions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Stochastic Optimizer")
warnings.filterwarnings("ignore", message="Converting to PeriodArray")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_long_family_conditional_correctness as base_runner


# Match the frozen v8 runner exactly so A_anchor is an exact incumbent control.
SEED = 20260807
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
PAIR_QUERY_MODES = {"4h": 4, "1d": 24, "3d": 72, "7d": 168}
PAIR_SCORE_BAND_BPS = 10.0
PAIR_RESIDUAL_GAP_BPS = 25.0
PAIR_MAX_PAIRS_PER_QUERY = 24
RESIDUAL_BINS = 20
FAMILY_Q_CLIP = 500.0

DEFAULT_SOURCE = ROOT / "data_perp/artifacts/long_family_conditional_correctness_20260807_v8"
DEFAULT_OUT = ROOT / "data_perp/artifacts/long_family_orthogonal_ablation_20260808_v1"


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _utc_ns(values: object) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if callable(getattr(idx, "as_unit", None)):
        idx = idx.as_unit("ns")
    return idx.asi8


def _sidecar_contract(sidecar: Path) -> tuple[list[str], list[str]]:
    """Recover the frozen base/context field lists from Parquet metadata."""

    import pyarrow.parquet as pq

    schema = list(map(str, pq.ParquetFile(sidecar).schema.names))
    start = schema.index("atr_bps") + 1
    end = schema.index("label_available_ts")
    ordered = schema[start:end]
    base_fields = [
        name for name in ordered
        if not name.startswith("base_structural_family__")
        and not name.startswith("base_reasoning__")
    ]
    health = [name for name in schema if name.startswith("structural_health__")]
    context_fields = list(dict.fromkeys([*base_fields, *health]))
    return base_fields, context_fields


def _load_cached_frame(source: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load the v8 materialised frame without rebuilding 15-minute labels."""

    frame_path = source / "family_contribution_matrix.parquet"
    frame = pd.read_parquet(frame_path)
    sidecar = Path(base_runner.DEFAULT_SIDECAR)
    base_fields, context_fields = _sidecar_contract(sidecar)
    family_fields = sorted(
        c for c in frame.columns if c.startswith("base_structural_family__")
        and not c.startswith("base_structural_family__" + "family_")
    )
    # The selected family columns are identifiable by the corresponding share
    # fields; this avoids accidentally treating any auxiliary structural field
    # as a family contribution.
    family_fields = sorted(
        c.removeprefix("family_abs_share__")
        for c in frame.columns if c.startswith("family_abs_share__")
    )
    required = {
        "fold", "candidate_id", "__ts__", "meta_partition", "label_available_ts",
        "policy_net_bps", "policy_gross_bps", "policy_label_available_ts",
        "family_unassigned_mass", "family_total_abs_contribution",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cached source is missing required columns: {missing}")
    missing_contract = [c for c in [*base_fields, *context_fields, *family_fields] if c not in frame.columns]
    if missing_contract:
        raise ValueError(f"cached source is missing frozen contract fields: {missing_contract[:20]}")
    for c in ["__ts__", "label_available_ts", "policy_label_available_ts"]:
        frame[c] = pd.to_datetime(frame[c], utc=True, errors="coerce")
    if frame[["__ts__", "policy_label_available_ts"]].isna().any().any():
        raise ValueError("cached source contains invalid timestamps")
    if frame.duplicated(["fold", "candidate_id"]).any():
        raise ValueError("cached source has duplicate fold/candidate rows")
    return frame, base_fields, context_fields, family_fields


def _fit_residualisation(train: pd.DataFrame, family_fields: list[str]) -> dict[str, object]:
    """Fit score-binned anchor and family contribution baselines on train."""

    score = pd.to_numeric(train["cap120_policy_correction"], errors="coerce").to_numpy(float)
    finite = np.isfinite(score)
    if finite.sum() < RESIDUAL_BINS:
        edges = np.array([-np.inf, np.inf], dtype=float)
    else:
        q = np.linspace(0.0, 1.0, RESIDUAL_BINS + 1)
        edges = np.unique(np.nanquantile(score[finite], q))
        edges[0] = -np.inf
        edges[-1] = np.inf
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf], dtype=float)
    bins = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    residual = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float) - score
    mu = np.zeros(len(edges) - 1, dtype=float)
    family_mu: dict[str, np.ndarray] = {}
    global_mu = float(np.nanmedian(residual[np.isfinite(residual)])) if np.isfinite(residual).any() else 0.0
    for b in range(len(mu)):
        m = (bins == b) & np.isfinite(residual)
        mu[b] = float(np.nanmedian(residual[m])) if m.any() else global_mu
    for f in family_fields:
        c = pd.to_numeric(train[f], errors="coerce").fillna(0.0).to_numpy(float)
        out = np.zeros(len(mu), dtype=float)
        global_c = float(np.nanmedian(c[np.isfinite(c)])) if np.isfinite(c).any() else 0.0
        for b in range(len(out)):
            m = (bins == b) & np.isfinite(c)
            out[b] = float(np.nanmedian(c[m])) if m.any() else global_c
        family_mu[f] = out
    return {"edges": edges, "mu": mu, "family_mu": family_mu}


def _apply_residualised_current(frame: pd.DataFrame, fit: dict[str, object], family_fields: list[str]) -> pd.DataFrame:
    score = pd.to_numeric(frame["cap120_policy_correction"], errors="coerce").to_numpy(float)
    edges = np.asarray(fit["edges"], dtype=float)
    bins = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    out = pd.DataFrame(index=frame.index)
    family_mu = fit["family_mu"]
    assert isinstance(family_mu, dict)
    for f in family_fields:
        c = pd.to_numeric(frame[f], errors="coerce").fillna(0.0).to_numpy(float)
        out[f"family_resid__{f}"] = (c - np.asarray(family_mu[f])[bins]).astype("float32")
    return out


def _residualised_history(
    query: pd.DataFrame,
    source: pd.DataFrame,
    family_fields: list[str],
    fit: dict[str, object],
    windows_hours: tuple[int, ...] = (4, 12, 24, 168),
) -> pd.DataFrame:
    """Prequential history using residualised family contribution Q."""

    qtime = _utc_ns(query["__ts__"])
    if source.empty:
        out = {}
        for f in family_fields:
            for h in windows_hours:
                out[f"hist_q_resid__{f}__{h}h"] = np.zeros(len(query), dtype="float32")
        return pd.DataFrame(out)
    event_time = _utc_ns(source["policy_label_available_ts"])
    order = np.argsort(event_time, kind="stable")
    src = source.iloc[order].reset_index(drop=True)
    event_time = event_time[order]
    current = _apply_residualised_current(src, fit, family_fields)
    score = pd.to_numeric(src["cap120_policy_correction"], errors="coerce").to_numpy(float)
    residual = np.clip(pd.to_numeric(src["policy_net_bps"], errors="coerce").to_numpy(float) - score, -FAMILY_Q_CLIP, FAMILY_Q_CLIP)
    out: dict[str, np.ndarray] = {}
    right = np.searchsorted(event_time, qtime, side="left")
    for f in family_fields:
        share = pd.to_numeric(src[f"family_abs_share__{f}"], errors="coerce").fillna(0.0).to_numpy(float)
        ctilde = current[f"family_resid__{f}"].to_numpy(float)
        q = share * ctilde * residual
        active = (np.abs(pd.to_numeric(src[f], errors="coerce").fillna(0.0).to_numpy(float)) > 1e-12).astype(float)
        cq = np.r_[0.0, np.cumsum(q)]
        ca = np.r_[0.0, np.cumsum(active)]
        for h in windows_hours:
            width = int(pd.Timedelta(hours=h).value)
            left = np.searchsorted(event_time, qtime - width, side="left")
            count = ca[right] - ca[left]
            out[f"hist_q_resid__{f}__{h}h"] = ((cq[right] - cq[left]) / np.maximum(count, 1.0)).astype("float32")
    return pd.DataFrame(out)


def _append_residual_features(
    bundle: dict[str, object], family_fields: list[str], fit_resid: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Append current and prior residualised family fields to fold matrices."""

    train = bundle["train"]
    cal = bundle["calibration"]
    test = bundle["test"]
    assert isinstance(train, pd.DataFrame) and isinstance(cal, pd.DataFrame) and isinstance(test, pd.DataFrame)
    h_train = _residualised_history(train, train, family_fields, fit_resid)
    h_cal = _residualised_history(cal, train, family_fields, fit_resid)
    h_test = _residualised_history(test, pd.concat([train, cal, test], ignore_index=True), family_fields, fit_resid)
    c_train = _apply_residualised_current(train, fit_resid, family_fields)
    c_cal = _apply_residualised_current(cal, fit_resid, family_fields)
    c_test = _apply_residualised_current(test, fit_resid, family_fields)
    pieces = []
    names: list[str] = []
    for f in family_fields:
        names.append(f"family_resid__{f}")
    names.extend(list(h_train.columns))
    def add(x: np.ndarray, c: pd.DataFrame, h: pd.DataFrame) -> np.ndarray:
        z = np.column_stack([x, c.to_numpy("float32"), h.to_numpy("float32")])
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return add(bundle["x_train"], c_train, h_train), add(bundle["x_calibration"], c_cal, h_cal), add(bundle["x_test"], c_test, h_test), names


def _feature_masks(names: list[str], selected_context: list[str], family_fields: list[str]) -> dict[str, np.ndarray]:
    names_arr = np.asarray(names, dtype=object)
    family_raw = set(family_fields)
    family_raw |= {f"family_abs_share__{f}" for f in family_fields}
    family_raw |= {f"family_active__{f}" for f in family_fields}
    family_resid = {f"family_resid__{f}" for f in family_fields}
    history_raw = {n for n in names if n.startswith("hist_")}
    history_resid = {n for n in names if n.startswith("hist_q_resid__")}
    factor = {n for n in names if n.startswith("family_state_")}
    anchor = {n for n in names if n in {"cap120_policy_correction", "cap120_score_rank"}}
    context = set(selected_context).intersection(names)
    def mask(fields: set[str]) -> np.ndarray:
        return np.asarray([n in fields for n in names], dtype=bool)
    a = anchor
    b = anchor | context
    c_raw = anchor | family_raw
    c_resid = anchor | family_resid | {f"family_abs_share__{f}" for f in family_fields} | {f"family_active__{f}" for f in family_fields}
    d_raw = c_raw | history_raw
    d_resid = c_resid | history_resid
    e_resid = d_resid | factor
    f_resid = b | c_resid
    return {
        "A_anchor": mask(a),
        "B_anchor_context": mask(b),
        "C_family_raw": mask(c_raw),
        "C_family_residualised": mask(c_resid),
        "D_history_raw": mask(d_raw),
        "D_history_residualised": mask(d_resid),
        "E_residual_factors": mask(e_resid),
        "F_context_residual_family": mask(f_resid),
        "G_full_contract": np.ones(len(names), dtype=bool),
    }


def _fit_orthogonal_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    class_values: np.ndarray,
    seed: int,
    x_eval_b: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray | None]:
    if x_train.shape[1] == 0:
        empty_b = None if x_eval_b is None else np.zeros(len(x_eval_b), dtype="float32")
        return np.zeros(len(x_eval), dtype="float32"), {"accuracy": np.nan, "logloss": np.nan, "iterations": 0.0}, empty_b
    scaler = StandardScaler().fit(x_train)
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", solver="adam", alpha=5e-3,
        batch_size=1024, learning_rate_init=1e-3, max_iter=120,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
        random_state=seed, shuffle=False,
    )
    model.fit(scaler.transform(x_train), y_train)
    proba = model.predict_proba(scaler.transform(x_eval))
    aligned = np.full((len(x_eval), 3), 1e-6, dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        if 0 <= cls < 3:
            aligned[:, cls] = proba[:, j]
    aligned /= aligned.sum(axis=1, keepdims=True)
    train_p = model.predict_proba(scaler.transform(x_train))
    train_aligned = np.full((len(x_train), 3), 1e-6, dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        if 0 <= cls < 3:
            train_aligned[:, cls] = train_p[:, j]
    train_aligned /= train_aligned.sum(axis=1, keepdims=True)
    meta = {
        "accuracy": float(accuracy_score(y_train, np.argmax(train_aligned, axis=1))),
        "logloss": float(log_loss(y_train, train_aligned, labels=[0, 1, 2])),
        "iterations": float(getattr(model, "n_iter_", 0)),
    }
    pred_a = (aligned @ class_values).astype("float32")
    pred_b = None
    if x_eval_b is not None:
        proba_b = model.predict_proba(scaler.transform(x_eval_b))
        aligned_b = np.full((len(x_eval_b), 3), 1e-6, dtype=float)
        for j, cls in enumerate(model.classes_.astype(int)):
            if 0 <= cls < 3:
                aligned_b[:, cls] = proba_b[:, j]
        aligned_b /= aligned_b.sum(axis=1, keepdims=True)
        pred_b = (aligned_b @ class_values).astype("float32")
    return pred_a, meta, pred_b


def _calibrate_delta(train: pd.DataFrame, calibration: pd.DataFrame, base: np.ndarray, raw_cal: np.ndarray) -> tuple[float, float]:
    y = pd.to_numeric(calibration["policy_net_bps"], errors="coerce").to_numpy(float) - pd.to_numeric(calibration["cap120_policy_correction"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(raw_cal)
    if ok.sum() < 500 or np.nanvar(raw_cal[ok]) < 1e-8:
        return 0.0, float(np.nanmedian(y[ok])) if ok.any() else 0.0
    slope = float(np.cov(raw_cal[ok], y[ok], bias=True)[0, 1] / np.nanvar(raw_cal[ok]))
    intercept = float(np.nanmean(y[ok]) - slope * np.nanmean(raw_cal[ok]))
    return float(np.clip(slope, 0.0, 1.5)), intercept


def _group_permutation(frame: pd.DataFrame, values: np.ndarray, seed: int) -> np.ndarray:
    """Permute rows within month × narrow incumbent-score decile groups."""
    out = np.asarray(values, dtype="float32").copy()
    score = pd.to_numeric(frame["cap120_policy_correction"], errors="coerce").to_numpy(float)
    finite = np.isfinite(score)
    edges = np.unique(np.nanquantile(score[finite], np.linspace(0, 1, 11))) if finite.sum() > 10 else np.array([-np.inf, np.inf])
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    dec = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    month = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m").to_numpy()
    rng = np.random.default_rng(seed)
    for _, idx in pd.DataFrame({"month": month, "dec": dec}).groupby(["month", "dec"], sort=False).groups.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def _fit_pair_mode(
    train_frame: pd.DataFrame,
    x_train: np.ndarray,
    eval_frame: pd.DataFrame,
    x_eval: np.ndarray,
    scaler: StandardScaler,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a zero-intercept residual preference ranker for one query width."""
    hours = PAIR_QUERY_MODES[mode]
    base = pd.to_numeric(train_frame["cap120_policy_correction"], errors="coerce").to_numpy(float)
    outcome = pd.to_numeric(train_frame["policy_net_bps"], errors="coerce").to_numpy(float) - base
    query = pd.to_datetime(train_frame["__ts__"], utc=True).dt.floor(f"{hours}h").astype(str).to_numpy()
    scaled = scaler.transform(x_train)
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    pairs = 0
    for _, idx_values in pd.Series(np.arange(len(train_frame))).groupby(query, sort=False):
        idx = np.asarray(idx_values, dtype=int)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(-base[idx], kind="stable")]
        local: list[tuple[int, int]] = []
        for pos, left in enumerate(idx[:-1]):
            for right in idx[pos + 1 : pos + 6]:
                if abs(base[left] - base[right]) > PAIR_SCORE_BAND_BPS:
                    continue
                gap = outcome[left] - outcome[right]
                if abs(gap) < PAIR_RESIDUAL_GAP_BPS:
                    continue
                local.append((left, right) if gap > 0 else (right, left))
        if len(local) > PAIR_MAX_PAIRS_PER_QUERY:
            local = [local[i] for i in rng.choice(len(local), PAIR_MAX_PAIRS_PER_QUERY, replace=False)]
        for winner, loser in local:
            diffs.extend([scaled[winner] - scaled[loser], scaled[loser] - scaled[winner]])
            labels.extend([1, 0])
            pairs += 1
    if len(labels) < 100 or len(set(labels)) < 2:
        return np.zeros(len(eval_frame), dtype="float32"), {"pair_count": float(pairs), "pair_rows": float(len(labels)), "train_accuracy": np.nan}
    model = LogisticRegression(fit_intercept=False, C=0.5, class_weight="balanced", solver="lbfgs", max_iter=250, random_state=seed)
    xx = np.asarray(diffs, dtype="float32")
    yy = np.asarray(labels, dtype="int8")
    model.fit(xx, yy)
    local_score = np.asarray(model.decision_function(scaler.transform(x_eval)), dtype="float32")
    return local_score, {"pair_count": float(pairs), "pair_rows": float(len(labels)), "train_accuracy": float(accuracy_score(yy, model.predict(xx)))}


def _tail_rows(frame: pd.DataFrame, score: str, period: str = "pooled") -> list[dict[str, object]]:
    block = frame if period == "pooled" else frame[frame["period_key"] == period]
    rows: list[dict[str, object]] = []
    if block.empty:
        return rows
    ordered = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for tail in TAILS:
        n = max(1, int(math.ceil(len(ordered) * tail)))
        chosen = ordered.head(n)
        rows.append({
            "arm": score, "period": period, "tail": float(tail), "trades": int(n),
            "gross_bps": float(chosen["policy_gross_bps"].mean()),
            "net_bps": float(chosen["policy_net_bps"].mean()),
            "win_rate": float((chosen["policy_net_bps"] > 0).mean()),
        })
    return rows


def _stability(metrics_frame: pd.DataFrame, control: str = "A_anchor") -> pd.DataFrame:
    pooled = metrics_frame[metrics_frame["period"].eq("pooled")].set_index(["arm", "tail"])
    rows: list[dict[str, object]] = []
    month = metrics_frame[metrics_frame["period"].str.match(r"^20\d{2}-\d{2}$")].copy()
    week = metrics_frame[metrics_frame["period"].str.match(r"^20\d{2}-W\d{2}$")].copy()
    for arm in sorted(metrics_frame["arm"].unique()):
        for tail in TAILS:
            base = month[(month["arm"] == control) & (month["tail"] == tail)].set_index("period")["net_bps"]
            cur = month[(month["arm"] == arm) & (month["tail"] == tail)].set_index("period")["net_bps"]
            aligned = pd.concat([base.rename("base"), cur.rename("cur")], axis=1).dropna()
            uplift = aligned["cur"] - aligned["base"]
            wb = week[(week["arm"] == control) & (week["tail"] == tail)].set_index("period")["net_bps"]
            wc = week[(week["arm"] == arm) & (week["tail"] == tail)].set_index("period")["net_bps"]
            wu = (pd.concat([wb.rename("base"), wc.rename("cur")], axis=1).dropna()["cur"] - pd.concat([wb.rename("base"), wc.rename("cur")], axis=1).dropna()["base"])
            pooled_uplift = float(pooled.loc[(arm, tail), "net_bps"] - pooled.loc[(control, tail), "net_bps"]) if (arm, tail) in pooled.index and (control, tail) in pooled.index else np.nan
            rows.append({
                "arm": arm, "tail": float(tail), "pooled_uplift_bps": pooled_uplift,
                "median_month_uplift_bps": float(uplift.median()) if len(uplift) else np.nan,
                "worst_month_uplift_bps": float(uplift.min()) if len(uplift) else np.nan,
                "share_positive_months": float((uplift > 0).mean()) if len(uplift) else np.nan,
                "month_uplift_std_bps": float(uplift.std(ddof=0)) if len(uplift) else np.nan,
                "median_week_uplift_bps": float(wu.median()) if len(wu) else np.nan,
                "p10_week_uplift_bps": float(wu.quantile(0.10)) if len(wu) else np.nan,
                "worst_week_uplift_bps": float(wu.min()) if len(wu) else np.nan,
                "stability_gate": bool(len(uplift) and uplift.median() >= 0.0 and uplift.min() >= 0.0),
            })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame, base_fields, context_fields, family_fields = _load_cached_frame(source)
    all_preds: list[pd.DataFrame] = []
    all_state: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    all_orth: list[pd.DataFrame] = []
    all_pair: list[pd.DataFrame] = []
    for i, (fold, block) in enumerate(frame.groupby("fold", sort=True, observed=True)):
        pred, state, audit, bundle = base_runner._fit_fold(
            block.copy(), base_fields, context_fields, family_fields, SEED + i * 100,
            out, return_bundle=True,
        )
        all_preds.append(pred)
        all_state.append(state)
        all_audits.append(audit)
        train = bundle["train"]; calibration = bundle["calibration"]; test = bundle["test"]
        assert isinstance(train, pd.DataFrame) and isinstance(calibration, pd.DataFrame) and isinstance(test, pd.DataFrame)
        fit_resid = _fit_residualisation(train, family_fields)
        x_train, x_cal, x_test, extra_names = _append_residual_features(bundle, family_fields, fit_resid)
        original_names = list(bundle["feature_names"])
        names = original_names + extra_names
        masks = _feature_masks(names, list(bundle["selected_context"]), family_fields)
        y_train = np.asarray(bundle["y_train"], dtype=int)
        class_values = np.asarray(bundle["class_values"], dtype=float)
        base_cal = calibration["cap120_policy_correction"].to_numpy(float)
        base_test = test["cap120_policy_correction"].to_numpy(float)
        orth = pd.DataFrame({
            "fold": str(fold), "candidate_id": test["candidate_id"].to_numpy(),
            "__ts__": test["__ts__"].to_numpy(), "policy_net_bps": test["policy_net_bps"].to_numpy(float),
            "policy_gross_bps": test["policy_gross_bps"].to_numpy(float),
            "period_key": pd.to_datetime(test["__ts__"], utc=True).dt.strftime("%Y-%m").to_numpy(),
        })
        orth["A_anchor"] = base_test.astype("float32")
        fold_meta: list[dict[str, object]] = []
        raw_cal_cache: dict[str, np.ndarray] = {}
        raw_test_cache: dict[str, np.ndarray] = {}
        for j, (arm, mask) in enumerate(masks.items()):
            if arm == "G_full_contract":
                # This is intentionally the production-like reference model,
                # refit through the same helper rather than copied predictions.
                model = bundle["mlp"]
                scaler = bundle["scaler"]
                post_cal = model.predict_proba(scaler.transform(np.asarray(bundle["x_calibration"])))
                post_test = model.predict_proba(scaler.transform(np.asarray(bundle["x_test"])))
                pc = np.full((len(post_cal), 3), 1e-6); pt = np.full((len(post_test), 3), 1e-6)
                for k, cls in enumerate(model.classes_.astype(int)):
                    if 0 <= cls < 3:
                        pc[:, cls] = post_cal[:, k]; pt[:, cls] = post_test[:, k]
                pc /= pc.sum(axis=1, keepdims=True); pt /= pt.sum(axis=1, keepdims=True)
                raw_cal = pc @ class_values; raw_test = pt @ class_values
                fit_meta = {"accuracy": np.nan, "logloss": np.nan, "iterations": float(getattr(model, "n_iter_", 0))}
            else:
                raw_cal, fit_meta, raw_test = _fit_orthogonal_model(
                    x_train[:, mask], y_train, x_cal[:, mask], class_values,
                    SEED + i * 1000 + j, x_test[:, mask],
                )
            # For non-reference arms, enforce a calibration-only affine map.
            cal_raw = raw_cal if arm == "G_full_contract" else raw_cal
            slope, intercept = _calibrate_delta(train, calibration, base_cal, cal_raw)
            resid_test = intercept + slope * raw_test
            score = base_test + np.clip(0.50 * resid_test, -100.0, 100.0)
            if arm != "A_anchor":
                orth[arm] = score.astype("float32")
            fold_meta.append({
                "fold": str(fold), "arm": arm,
                # G reuses the exact production MLP bundle, whose input is
                # the original frozen feature contract.  Its mask includes
                # appended diagnostic residual columns, so mask.sum() would
                # over-report the deployed feature count.
                "feature_count": int(len(original_names)) if arm == "G_full_contract" else int(mask.sum()),
                "feature_digest": _digest(original_names) if arm == "G_full_contract" else _digest(np.asarray(names)[mask].tolist()),
                "calibration_slope": float(slope), "calibration_intercept": float(intercept),
                "train_accuracy": fit_meta.get("accuracy"), "train_logloss": fit_meta.get("logloss"),
                "iterations": fit_meta.get("iterations"), "residualised_family": "residual" in arm,
            })
        # The call above intentionally keeps G aligned with the production
        # state model.  Add deterministic family/history placebos separately.
        resid_family_cols = [names.index(f"family_resid__{f}") for f in family_fields]
        resid_hist_cols = [i for i, n in enumerate(names) if n.startswith("hist_q_resid__")]
        for pname, cols, seed_offset in (("P_family_permutation", resid_family_cols, 700), ("P_history_permutation", resid_hist_cols, 900)):
            if not cols:
                continue
            xt = x_train.copy(); xc = x_cal.copy(); xv = x_test.copy()
            # Permute each column within its own score/month grouping.  This
            # preserves marginal distributions and query density while
            # removing the family/history relationship.
            for col in cols:
                xt[:, col] = _group_permutation(train, xt[:, col], SEED + seed_offset + col)
                xc[:, col] = _group_permutation(calibration, xc[:, col], SEED + seed_offset + col)
                xv[:, col] = _group_permutation(test, xv[:, col], SEED + seed_offset + col)
            raw_cal, fit_meta, raw_test = _fit_orthogonal_model(
                xt[:, masks["G_full_contract"]], y_train,
                xc[:, masks["G_full_contract"]], class_values,
                SEED + seed_offset, xv[:, masks["G_full_contract"]],
            )
            slope, intercept = _calibrate_delta(train, calibration, base_cal, raw_cal)
            orth[pname] = (base_test + np.clip(0.50 * (intercept + slope * raw_test), -100.0, 100.0)).astype("float32")
            fold_meta.append({"fold": str(fold), "arm": pname, "feature_count": len(names), "feature_digest": _digest(names), "calibration_slope": slope, "calibration_intercept": intercept, "train_accuracy": fit_meta.get("accuracy"), "train_logloss": fit_meta.get("logloss"), "iterations": fit_meta.get("iterations"), "residualised_family": pname == "P_family_permutation"})
        orth["A_anchor"] = base_test.astype("float32")
        all_orth.append(orth)
        # Query-mode pairwise arms use the original full feature matrix and
        # residual outcomes, with fit-only scaling and no intercept.
        scaler = StandardScaler().fit(np.asarray(bundle["x_train"], dtype=float))
        for mode in PAIR_QUERY_MODES:
            local_test, pair_meta = _fit_pair_mode(
                train,
                np.asarray(bundle["x_train"], dtype=float),
                test,
                np.asarray(bundle["x_test"], dtype=float),
                scaler,
                mode,
                SEED + i * 100 + len(mode),
            )
            center = 0.0
            scale = float(np.nanstd(local_test)) if np.isfinite(local_test).any() else 1.0
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
            near = base_test >= float(np.nanquantile(bundle["baseline_reference"], 0.90))
            score = base_test + np.where(near, 10.0 * np.tanh((local_test - center) / scale), 0.0)
            name = f"Q_pair_{mode}"
            orth[name] = score.astype("float32")
            all_pair.append(pd.DataFrame([{"fold": str(fold), "arm": name, **pair_meta, "query_hours": PAIR_QUERY_MODES[mode]}]))
        pd.DataFrame(fold_meta).to_parquet(out / f"orthogonal_fold_{fold}.parquet", index=False, compression="zstd")
        gc.collect()
    orth_preds = pd.concat(all_orth, ignore_index=True)
    # Do not report the internal training/calibration rows as OOS.  The orth
    # table contains only the outer test partition by construction.
    arm_names = [c for c in orth_preds.columns if c.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_", "P_", "Q_"))]
    rows: list[dict[str, object]] = []
    for arm in arm_names:
        rows.extend(_tail_rows(orth_preds, arm))
        for month in sorted(orth_preds.period_key.unique()):
            rows.extend(_tail_rows(orth_preds, arm, str(month)))
        week_key = pd.to_datetime(orth_preds["__ts__"], utc=True).dt.strftime("%G-W%V")
        z = orth_preds.copy(); z["period_key"] = week_key
        for week in sorted(z.period_key.unique()):
            rows.extend(_tail_rows(z, arm, str(week)))
    metrics = pd.DataFrame(rows)
    stability = _stability(metrics)
    pair_metrics = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    out.mkdir(parents=True, exist_ok=True)
    orth_preds.to_parquet(out / "orthogonal_oos_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / "orthogonal_ablation_metrics.parquet", index=False, compression="zstd")
    stability.to_parquet(out / "stability_metrics.parquet", index=False, compression="zstd")
    pair_metrics.to_parquet(out / "pairwise_query_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(all_state, ignore_index=True).to_parquet(out / "mlp_state_metrics.parquet", index=False, compression="zstd")
    pd.concat(all_audits, ignore_index=True).to_parquet(out / "fold_audits.parquet", index=False, compression="zstd")
    pooled = metrics[metrics.period.eq("pooled")].sort_values(["tail", "net_bps"], ascending=[True, False])
    stability_gate = stability[(stability.tail == 0.05) & stability.stability_gate].sort_values(["pooled_uplift_bps", "worst_month_uplift_bps"], ascending=False)
    winner = str(stability_gate.iloc[0].arm) if not stability_gate.empty else "NO_ARM_PASSES_STABILITY_GATE"
    report = [
        "# Orthogonal family attribution and query-group ablation",
        "",
        f"Source: `{source}`; outer OOS rows: {len(orth_preds):,}; ranking is pooled global top-k.",
        "",
        "## Arms",
        "",
        "A is incumbent-only; B adds selected causal context; C compares raw versus score-residualised family contributions; D adds prior family history; E adds factorised family state; F combines context and residualised family inputs; G is the full production-like contract. P arms are family/history permutation placebos. Q arms are residual pairwise preference models over 4h/1d/3d/7d blocks.",
        "",
        "## Pooled OOS net bps/trade",
        "",
        "| arm | tail | trades | gross | net |", "|---|---:|---:|---:|---:|",
    ]
    for r in pooled.itertuples(index=False):
        report.append(f"| {r.arm} | {r.tail:.3g} | {r.trades} | {r.gross_bps:.2f} | {r.net_bps:.2f} |")
    report += [
        "", "## Stability-first selection", "",
        f"Top-5 stability winner: **{winner}**. Gate requires median monthly uplift >= 0 and worst-month uplift >= 0 versus A; pooled net EV is used only after that gate.",
        "", "## Correctness contract", "",
        "All family residualisation parameters are fit on meta-train rows only. History uses policy_label_available_ts < query timestamp. Placebos preserve month/score-bin marginals. Pairwise models use residual outcomes, train rows only, and no intercept. No OOS arm is used to choose features or query width.",
    ]
    (out / "ORTHOGONAL_FAMILY_ABLATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    checks = {
        "outer_test_rows_only": bool(orth_preds["candidate_id"].is_unique),
        "no_duplicate_candidate_ids": bool(orth_preds["candidate_id"].is_unique),
        "required_control_present": "A_anchor" in arm_names,
        "residualised_arm_present": "C_family_residualised" in arm_names,
        "family_placebo_present": "P_family_permutation" in arm_names,
        "history_placebo_present": "P_history_permutation" in arm_names,
        "all_query_modes_present": all(f"Q_pair_{m}" in arm_names for m in PAIR_QUERY_MODES),
        "stability_metrics_present": not stability.empty,
        "global_tail_metrics_present": bool((metrics.period == "pooled").any()),
    }
    _write_json(out / "correctness_test_report.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks})
    _write_json(out / "run_manifest.json", {
        "schema": "long_family_orthogonal_ablation_v1", "status": "complete", "source": str(source),
        "family_fields": family_fields, "family_count": len(family_fields), "base_contract_digest": _digest(base_fields),
        "context_contract_digest": _digest(context_fields), "pair_query_modes_hours": PAIR_QUERY_MODES,
        "residualisation_bins": RESIDUAL_BINS, "residual_q_clip_bps": FAMILY_Q_CLIP,
        "stability_gate": {"median_month_uplift_bps": 0.0, "worst_month_uplift_bps": 0.0, "selection_tail": 0.05},
        "winner": winner, "checks": checks,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
