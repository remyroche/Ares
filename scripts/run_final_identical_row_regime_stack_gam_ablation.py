#!/usr/bin/env python3
"""Fail-closed final identical-row base/residual/GAM regime-stack ablation.

This runner is intentionally inert until an authoritative manifest supplies
sealed 2022--2025 OOF and frozen 2026-forward soft sidecars under the contract
below.  It never treats raw/rejected state IDs or fold-local posterior axes as
features.  All learnt arms are side-local; every arm is fit and causally EV
mapped on pre-2026 rows only, then assessed once on identical 2026 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, validate_candidate_identity

SCHEMA = "final_identical_row_regime_stack_gam_ablation_v3"
SIDECAR_SCHEMA = "authoritative_soft_regime_transition_sidecars_v1"
OUT = ROOT / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3"
HISTORICAL_LEDGER = ROOT / "data_perp/artifacts/frozen_contextual_score_arms_2023apr_2025jun_20260730_v1/blocked_oof_training_panel.parquet"
FORWARD_LEDGER = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet"
TARGET, GROSS, COST, ALPHA = "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__first_touch_target_soft__"
BASE, RESIDUAL = "score_base_alpha", "score_residual_expected_ev"
# The authority publishes hourly tables, not candidate-level context.  These
# are deliberately semantic continuous values only: neither raw GMM/cluster
# identities nor posterior axes can enter this experiment.
REGIME_SOURCE = {
    "regime_change_probability_mean": "bocpd__change_probability_mean",
    "regime_change_probability_max": "bocpd__change_probability_max",
    "regime_run_length_mean": "bocpd__run_length_mean",
    "regime_run_length_q05": "bocpd__run_length_q05",
    "regime_run_length_entropy": "bocpd__run_length_entropy",
    "regime_signal_count": "bocpd__signal_count",
    "regime_state_age_hours": "bocpd__state_age_hours",
    "regime_is_persistent_24h": "bocpd__is_persistent_24h",
    "regime_is_persistent_72h": "bocpd__is_persistent_72h",
}
TRANSITION_SOURCE = {
    "transition_lgbm_probability": "lgbm_transition_probability",
    "transition_lgbm_entropy": "lgbm_entropy",
    "transition_lgbm_margin": "lgbm_margin",
    "transition_bocpd_stable_probability": "bocpd_stable_vs_transition_probability",
    "transition_bocpd_onset_h1_probability": "bocpd_onset_h1_probability",
    "transition_bocpd_onset_h3_probability": "bocpd_onset_h3_probability",
    "transition_bocpd_onset_h6_probability": "bocpd_onset_h6_probability",
    "transition_bocpd_onset_h12_probability": "bocpd_onset_h12_probability",
}
REGIME = tuple(REGIME_SOURCE)
TRANSITION = tuple(TRANSITION_SOURCE)
TOP = .10


@dataclass(frozen=True)
class Arm:
    name: str
    placement: str
    context: str
    target: str
    family: str


ARMS = (
    Arm("baseline", "baseline", "none", TARGET, "raw"),
    Arm("base_regime_only", "base", "regime", ALPHA, "lgbm"),
    Arm("base_transition_only", "base", "transition", ALPHA, "lgbm"),
    Arm("base_combined", "base", "combined", ALPHA, "lgbm"),
    Arm("residual_regime_only", "residual_trust", "regime", TARGET, "lgbm"),
    Arm("residual_transition_only", "residual_trust", "transition", TARGET, "lgbm"),
    Arm("residual_combined", "residual_trust", "combined", TARGET, "lgbm"),
    Arm("gam_regime_only", "additive_bounded_gam", "regime", TARGET, "gam"),
    Arm("gam_transition_only", "additive_bounded_gam", "transition", TARGET, "gam"),
    Arm("gam_combined", "additive_bounded_gam", "combined", TARGET, "gam"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def fields(arm: Arm) -> list[str]:
    base = [BASE] if arm.placement == "base" else [RESIDUAL]
    if arm.context == "regime": return [*base, *REGIME]
    if arm.context == "transition": return [*base, *TRANSITION]
    if arm.context == "combined": return [*base, *REGIME, *TRANSITION]
    return base


def _load_manifest(path: Path) -> tuple[dict[str, Any], Path, Path]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SIDECAR_SCHEMA or not str(payload.get("status", "")).startswith("SEALED"):
        raise RegimeOOFStackError("authoritative sealed soft-sidecar manifest is required")
    marker = path.with_name("manifest.sha256")
    if not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != sha(path):
        raise RegimeOOFStackError("soft-sidecar manifest checksum is missing or invalid")
    if payload.get("model_sample_cadence") != "1h" or payload.get("assessment_sample_cadence") != "1h":
        raise RegimeOOFStackError("soft-sidecar manifest does not prove 1h training and assessment")
    historical_contract, forward_contract = str(payload.get("historical_contract", "")), str(payload.get("forward_contract", ""))
    if "blocked-OOF" not in historical_contract or "untouched 2026" not in forward_contract:
        raise RegimeOOFStackError("soft-sidecar manifest does not prove the required 2022--2025/2026 split")
    checksums = payload.get("outputs_sha256", {})
    paths = tuple(path.parent / name for name in ("soft_regime_hourly.parquet", "soft_transition_hourly.parquet"))
    for sidecar in paths:
        if not sidecar.is_file() or checksums.get(sidecar.name) != sha(sidecar):
            raise RegimeOOFStackError(f"soft-sidecar checksum mismatch: {sidecar}")
    return payload, *paths


def _hourly(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "source_utc" not in frame:
        raise RegimeOOFStackError(f"{name} lacks source_utc")
    result = frame.copy()
    result["source_utc"] = pd.to_datetime(result["source_utc"], utc=True, errors="raise")
    if result.source_utc.duplicated().any() or (result.source_utc.astype("int64") % pd.Timedelta(hours=1).value != 0).any():
        raise RegimeOOFStackError(f"{name} is not exactly one row per 1h timestamp")
    forbidden = [field for field in result if any(token in field.lower() for token in ("state_id", "state_p_raw", "gmm", "morphology", "execution_net", "execution_gross", "future_", "mfe", "mae", "timing", "wait"))]
    if forbidden:
        raise RegimeOOFStackError(f"raw/rejected state identity or outcome leaked into {name}: {forbidden[:8]}")
    return result


def _hourly_context(regime_path: Path, transition_path: Path) -> pd.DataFrame:
    regime = _hourly(pd.read_parquet(regime_path), name="regime hourly sidecar")
    transition = _hourly(pd.read_parquet(transition_path), name="transition hourly sidecar")
    shared = [field for field in regime.columns if field != "source_utc" and field in transition]
    for field in shared:
        if not regime[field].equals(transition[field]):
            raise RegimeOOFStackError(f"hourly sidecars disagree on shared field {field}")
    combined = regime.merge(transition.drop(columns=shared), on="source_utc", how="outer", validate="one_to_one")
    required = [*REGIME_SOURCE.values(), *TRANSITION_SOURCE.values(), "bocpd_regime_available", "lgbm_transition_available", "provenance_partition_bocpd", "provenance_partition_lgbm", "train_end_exclusive_utc_bocpd", "train_end_exclusive_utc_lgbm", "fit_label_resolution_max_utc_bocpd", "fit_label_resolution_max_utc_lgbm"]
    missing = [field for field in required if field not in combined]
    if missing:
        raise RegimeOOFStackError(f"authoritative hourly sidecars are missing {missing}")
    result = combined.loc[:, ["source_utc", *REGIME_SOURCE.values(), *TRANSITION_SOURCE.values(), "bocpd_regime_available", "lgbm_transition_available", "provenance_partition_bocpd", "provenance_partition_lgbm", "train_end_exclusive_utc_bocpd", "train_end_exclusive_utc_lgbm", "fit_label_resolution_max_utc_bocpd", "fit_label_resolution_max_utc_lgbm"]].rename(columns={**{source: target for target, source in REGIME_SOURCE.items()}, **{source: target for target, source in TRANSITION_SOURCE.items()}})
    return _hourly(result, name="joined authoritative hourly context")


def _scores(path: Path) -> pd.DataFrame:
    columns = [*IDENTITY_COLUMNS, "execution_label_end_utc", TARGET, GROSS, COST, ALPHA, BASE, RESIDUAL]
    aliases = {
        "execution_label_end_utc": "execution_label_available_at",
        BASE: "base_oof_score",
        RESIDUAL: "residual_expected_ev",
    }
    available = set(pq.read_schema(path).names)
    optional = [alias for alias in aliases.values() if alias in available]
    work = validate_candidate_identity(pd.read_parquet(path, columns=[*columns, *optional]))
    for target, alias in aliases.items():
        if alias not in work:
            continue
        if target == "execution_label_end_utc":
            left = pd.to_datetime(work[target], utc=True, errors="coerce")
            right = pd.to_datetime(work[alias], utc=True, errors="coerce")
            conflict = left.notna() & right.notna() & left.ne(right)
            work[target] = left.fillna(right)
        else:
            left = pd.to_numeric(work[target], errors="coerce")
            right = pd.to_numeric(work[alias], errors="coerce")
            conflict = left.notna() & right.notna() & ~np.isclose(left, right, atol=1e-12, rtol=0)
            work[target] = left.fillna(right)
        if conflict.any():
            raise RegimeOOFStackError(f"score-ledger aliases conflict for {target} and {alias}")
        work.drop(columns=alias, inplace=True)
    work.execution_label_end_utc = pd.to_datetime(work.execution_label_end_utc, utc=True, errors="raise")
    if not (work.__ts__.dt.minute.eq(0) & work.__ts__.dt.second.eq(0)).all(): raise RegimeOOFStackError("score ledger is not hourly")
    if work.execution_label_end_utc.isna().any() or (work.execution_label_end_utc <= work.__ts__).any(): raise RegimeOOFStackError("score ledger label availability invalid")
    if not np.allclose(work[GROSS] - work[COST], work[TARGET], atol=1e-10, rtol=0): raise RegimeOOFStackError("gross-cost-net identity failed")
    return work


def _verified_scores(path: Path, *, role: str) -> pd.DataFrame:
    """Accept only score ledgers with a sealed, matching lineage when present.

    Unit fixtures deliberately have no parent manifest.  A materialized ledger
    does, and must prove both its checksum and the specific OOF/forward
    contract before it can be combined with the hourly sidecars.
    """
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return _scores(path)
    marker = path.parent / "manifest.sha256"
    if not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != sha(manifest_path):
        raise RegimeOOFStackError(f"{role} score-ledger manifest checksum is missing or invalid")
    manifest = json.loads(manifest_path.read_text())
    # A temporary sidecar manifest can share a directory with a unit fixture;
    # it is not a score-ledger lineage document.  Real score ledgers must use
    # one of the two explicit schemas below.
    if manifest.get("schema") not in {"frozen_contextual_score_arms_v1", "mayjul2026_exact_allscore_ic_ev_waterfall_v1"}:
        return _scores(path)
    if role == "historical":
        contract = str(manifest.get("frozen_contextual_coefficients", {}).get("blocked_oof_requirement", ""))
        expected_hash = manifest.get("outputs", {}).get(path.name)
        if manifest.get("schema") != "frozen_contextual_score_arms_v1" or "held blocked OOF" not in contract or "labels resolve before freeze" not in contract:
            raise RegimeOOFStackError("historical score ledger does not prove strict blocked-OOF score lineage")
    elif role == "forward":
        contract = str(manifest.get("contracts", {}).get("oof", ""))
        expected_hash = manifest.get("outputs", {}).get("allscore_waterfall", {}).get("sha256")
        if manifest.get("schema") != "mayjul2026_exact_allscore_ic_ev_waterfall_v1" or "strict prior-resolved side-local OOF" not in contract:
            raise RegimeOOFStackError("forward score ledger does not prove strict side-local OOF lineage")
    else:
        raise RegimeOOFStackError(f"unknown score-ledger role {role}")
    if expected_hash != sha(path):
        raise RegimeOOFStackError(f"{role} score-ledger checksum mismatch")
    return _scores(path)


def _join(scores: pd.DataFrame, context: pd.DataFrame, *, role: str) -> pd.DataFrame:
    """Join the hourly authority many-to-one without changing score-ledger IDs."""
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    if role == "historical":
        if not scores.__ts__.lt(cutoff).all():
            raise RegimeOOFStackError("historical score ledger contains 2026 candidates")
        expected_partition = "blocked_oof_2022_2025"
    elif role == "forward":
        if not scores.__ts__.ge(cutoff).all():
            raise RegimeOOFStackError("forward score ledger contains pre-2026 candidates")
        expected_partition = "untouched_2026_forward"
    else:
        raise RegimeOOFStackError(f"unknown score/context role {role}")
    before = scores.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    joined = scores.merge(context, left_on="__ts__", right_on="source_utc", how="left", validate="many_to_one", sort=False)
    after = joined.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    if len(joined) != len(scores) or not before.equals(after) or not validate_candidate_identity(joined).loc[:, list(IDENTITY_COLUMNS)].equals(joined.loc[:, list(IDENTITY_COLUMNS)]):
        raise RegimeOOFStackError(f"{role} hourly join changed exact candidate identity")
    required = [*REGIME, *TRANSITION]
    if joined.source_utc.isna().any():
        raise RegimeOOFStackError(f"{role} score ledger has timestamps absent from the sealed hourly authority")
    available = (
        joined[required].notna().all(axis=1)
        & joined.bocpd_regime_available.fillna(False).astype(bool)
        & joined.lgbm_transition_available.fillna(False).astype(bool)
    )
    if role == "forward" and not available.all():
        raise RegimeOOFStackError("forward score ledger includes unavailable hourly context")
    if role == "historical" and (~available).any():
        warmup = joined.loc[~available]
        for suffix in ("bocpd", "lgbm"):
            partition = f"provenance_partition_{suffix}"
            if not warmup[partition].eq("blocked_oof_warmup_unavailable").all():
                raise RegimeOOFStackError(
                    f"historical context is missing outside the explicit {suffix} OOF warm-up"
                )
        # Freeze one common context-available universe for every arm.  The
        # excluded warm-up count is reported separately; no value is imputed.
        joined = joined.loc[available].copy()
    for suffix in ("bocpd", "lgbm"):
        partition = f"provenance_partition_{suffix}"
        if not joined[partition].eq(expected_partition).all():
            raise RegimeOOFStackError(f"{role} hourly context provenance is not {expected_partition} for {suffix}")
        if role == "historical":
            train_end = pd.to_datetime(joined[f"train_end_exclusive_utc_{suffix}"], utc=True, errors="raise")
            resolved = pd.to_datetime(joined[f"fit_label_resolution_max_utc_{suffix}"], utc=True, errors="raise")
            if train_end.isna().any() or resolved.isna().any() or not resolved.lt(train_end).all():
                raise RegimeOOFStackError(f"{suffix} historical hourly context fails strict resolved-label OOF")
    joined.attrs["input_score_rows"] = len(scores)
    joined.attrs["excluded_explicit_warmup_rows"] = len(scores) - len(joined)
    return joined.drop(columns=["source_utc"])


def _matrix(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    z = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median().fillna(0.)
    return x.fillna(med).to_numpy(np.float32), z.fillna(med).to_numpy(np.float32)


def _predict(train: pd.DataFrame, test: pd.DataFrame, arm: Arm, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if arm.family == "raw": return test[RESIDUAL].to_numpy(float), {"family":"frozen_residual_score"}
    target = pd.to_numeric(train[arm.target], errors="coerce")
    fit = train.loc[target.notna()].copy()
    if len(fit) < 8:
        raise RegimeOOFStackError(f"{arm.name} has insufficient non-null {arm.target} fit rows")
    x, z = _matrix(fit, test, fields(arm)); y = pd.to_numeric(fit[arm.target], errors="raise").to_numpy(float)
    if arm.family == "lgbm":
        model = lgb.LGBMRegressor(n_estimators=160, learning_rate=.035, num_leaves=15, min_child_samples=180, subsample=.85, colsample_bytree=.9, reg_lambda=4., random_state=seed, n_jobs=4, verbosity=-1).fit(x, y)
        return np.asarray(model.predict(z), float), {"family":"side_local_lgbm","target":arm.target,"features":fields(arm),"non_null_target_fit_rows":len(fit),"excluded_null_target_fit_rows":len(train)-len(fit)}
    model = Pipeline([("splines", SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)), ("ridge", Ridge(alpha=2.0))]).fit(x, y)
    raw = np.asarray(model.predict(z), float); train_raw = np.asarray(model.predict(x), float)
    bound = np.quantile(train_raw, [.01, .99]); return np.clip(raw, *bound), {"family":"additive_bounded_spline_gam","target":arm.target,"features":fields(arm),"bounds":bound.tolist(),"non_null_target_fit_rows":len(fit),"excluded_null_target_fit_rows":len(train)-len(fit)}


def _mapper(scores: np.ndarray, labels: np.ndarray):
    valid = np.isfinite(scores) & np.isfinite(labels); scores, labels = scores[valid], labels[valid]
    if len(scores) < 8 or np.unique(scores).size < 2:
        value = float(labels.mean()) if len(labels) else 0.; return lambda x: np.full(len(x), value)
    # Every upstream score is explicitly oriented so that larger means better.
    # Auto-direction may reverse that economic contract on a noisy OOF window.
    model = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(scores, labels)
    return lambda x: np.asarray(model.predict(np.asarray(x, float)), float)


def _oof(history: pd.DataFrame, arm: Arm, *, start: pd.Timestamp, min_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; audit=[]
    blocks=pd.date_range(start.normalize(), history.__ts__.max().normalize()+pd.Timedelta(days=1), freq="3MS", tz="UTC")
    for number, block in enumerate(blocks):
        evaluation=history.loc[(history.__ts__ >= block)&(history.__ts__ < block+pd.DateOffset(months=3))].copy()
        training=history.loc[history.execution_label_end_utc < block].copy()
        if evaluation.empty or len(training)<min_train: continue
        for side, local in evaluation.groupby("side_name", observed=True):
            fit=training.loc[training.side_name.eq(side)]
            if len(fit)<min_train//3: continue
            if arm.family != "raw" and pd.to_numeric(fit[arm.target],errors="coerce").notna().sum()<8: continue
            raw, model=_predict(fit,local,arm,9100+number*37+(side=="short")); rows.append(local.loc[:,list(IDENTITY_COLUMNS)+["execution_label_end_utc",TARGET]].assign(arm=arm.name,oof_block_start_utc=block,raw_score=raw));audit.append({"arm":arm.name,"placement":arm.placement,"side_name":side,"oof_block_start_utc":block,"train_rows":len(fit),"evaluation_rows":len(local),"train_label_end_max":fit.execution_label_end_utc.max(),**model})
    if not rows: raise RegimeOOFStackError(f"no prior OOF support for {arm.name}")
    return validate_candidate_identity(pd.concat(rows,ignore_index=True)),pd.DataFrame(audit)


def _rank(left: pd.Series, right: pd.Series) -> float:
    valid=left.notna()&right.notna()
    return float(left.loc[valid].rank().corr(right.loc[valid].rank())) if valid.sum()>=3 else float("nan")


def _evaluate(frame: pd.DataFrame, arm: Arm) -> tuple[dict[str,Any],pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # Selection remains one pooled global top-k after causal EV mapping.  Raw
    # score is used only inside exact mapped-EV ties, never ahead of the map.
    ordered=frame.sort_values(["mapped_score","raw_score","candidate_id"],ascending=[False,False,True],kind="stable"); selected=ordered.head(max(1,math.ceil(len(frame)*TOP))).copy();frame=frame.copy();frame["selected_global_top10"]=frame.candidate_id.isin(set(selected.candidate_id))
    periods=[]
    for kind,key in (("week",frame.__ts__.dt.strftime("%G-W%V")),("month",frame.__ts__.dt.strftime("%Y-%m"))):
        for period,local in frame.groupby(key,observed=True,sort=True):
            pick=local.loc[local.selected_global_top10];periods.append({"arm":arm.name,"period_type":kind,"period":period,"candidate_rows":len(local),"global_selected_rows":len(pick),"alpha_rank_ic":_rank(local.mapped_score,local[ALPHA]),"execution_rank_ic":_rank(local.mapped_score,local[TARGET]),"raw_score_net_rank_ic":_rank(local.raw_score,local[TARGET]),"mean_mapped_ev":pick.mapped_score.mean(),"mean_net_ev":pick[TARGET].mean(),"mean_gross_ev":pick[GROSS].mean(),"mean_cost":pick[COST].mean(),"hit_rate":pick[TARGET].gt(0).mean()})
    p=pd.DataFrame(periods);summary={"arm":arm.name,"placement":arm.placement,"context":arm.context,"family":arm.family,"candidate_rows":len(frame),"top10_rows":len(selected),"alpha_rank_ic":_rank(frame.mapped_score,frame[ALPHA]),"execution_rank_ic":_rank(frame.mapped_score,frame[TARGET]),"raw_score_net_rank_ic":_rank(frame.raw_score,frame[TARGET]),"mapped_ev_mean":frame.mapped_score.mean(),"top10_net_ev":selected[TARGET].mean(),"top10_gross_ev":selected[GROSS].mean(),"top10_cost":selected[COST].mean(),"top10_hit_rate":selected[TARGET].gt(0).mean()}
    for kind in ("week","month"):
        q=p.loc[p.period_type.eq(kind)];summary[f"{kind}_ic_q10"],summary[f"{kind}_ic_q50"]=q.alpha_rank_ic.quantile(.1),q.alpha_rank_ic.quantile(.5);summary[f"{kind}_net_ev_q10"],summary[f"{kind}_net_ev_q50"]=q.mean_net_ev.quantile(.1),q.mean_net_ev.quantile(.5)
        latest=q.sort_values("period").tail(1).iloc[0];worst=q.loc[q.mean_net_ev.idxmin()];summary[f"latest_{kind}"],summary[f"latest_{kind}_net_ev"],summary[f"worst_{kind}"],summary[f"worst_{kind}_net_ev"]=latest.period,latest.mean_net_ev,worst.period,worst.mean_net_ev
    sides=[]
    for side,local in frame.groupby("side_name",observed=True):
        pick=local.loc[local.selected_global_top10];assets=pick.__symbol__.value_counts(normalize=True);sides.append({"arm":arm.name,"side_name":side,"candidate_rows":len(local),"global_selected_rows":len(pick),"alpha_rank_ic":_rank(local.mapped_score,local[ALPHA]),"execution_rank_ic":_rank(local.mapped_score,local[TARGET]),"top10_net_ev":pick[TARGET].mean(),"top10_hit_rate":pick[TARGET].gt(0).mean(),"asset_hhi":(assets**2).sum() if len(assets) else np.nan})
    recall=[]
    for event,mask in (("positive_net",frame[TARGET].gt(0)),("gross_exceeds_cost",frame[GROSS].gt(frame[COST]))): recall.append({"arm":arm.name,"event":event,"population_rows":int(mask.sum()),"selected_rows":int(frame.loc[mask,"selected_global_top10"].sum()),"recall":float(frame.loc[mask,"selected_global_top10"].mean()) if mask.any() else np.nan})
    calibration=frame.loc[:,["mapped_score",TARGET]].copy();calibration["bin"]=pd.qcut(calibration.mapped_score.rank(method="first"),q=10,labels=False,duplicates="drop");calibration=calibration.groupby("bin",observed=True).agg(candidate_rows=(TARGET,"size"),mean_mapped_ev=("mapped_score","mean"),mean_net_ev=(TARGET,"mean")).reset_index();calibration["arm"]=arm.name;calibration["signed_error"]=calibration.mean_mapped_ev-calibration.mean_net_ev;summary["calibration_mae_decile"]=calibration.signed_error.abs().mean();summary["calibration_bias_decile"]=calibration.signed_error.mean();ties=frame.mapped_score.value_counts(dropna=False);summary["mapped_tie_mass"]=ties.max()/len(frame);assets=selected.__symbol__.value_counts(normalize=True);summary["selected_asset_hhi"],summary["selected_largest_asset_share"],summary["selected_long_share"]=(assets**2).sum(),assets.iloc[0],selected.side_name.eq("long").mean()
    return summary,p,pd.DataFrame(sides),pd.DataFrame(recall),calibration


def run(*, sidecar_manifest: Path, historical_scores: Path, current_scores: Path, output: Path=OUT, oof_start: str="2023-01-01T00:00:00Z", min_train_rows:int=12000, max_map_age_days:int=90) -> Path:
    output=Path(output)
    if output.exists(): raise RegimeOOFStackError(f"refusing to overwrite {output}")
    manifest,regime_path,transition_path=_load_manifest(Path(sidecar_manifest));context=_hourly_context(regime_path, transition_path)
    historical_ledger=_verified_scores(Path(historical_scores),role="historical");forward_ledger=_verified_scores(Path(current_scores),role="forward")
    history=_join(historical_ledger,context,role="historical");current=_join(forward_ledger,context,role="forward")
    intersection_coverage=pd.DataFrame([
        {"partition":"historical","input_score_rows":len(historical_ledger),"common_context_rows":len(history),"excluded_explicit_warmup_rows":len(historical_ledger)-len(history)},
        {"partition":"forward","input_score_rows":len(forward_ledger),"common_context_rows":len(current),"excluded_explicit_warmup_rows":len(forward_ledger)-len(current)},
    ])
    input_score_coverage=pd.DataFrame([
        {"partition":name,"rows":len(frame),"label_end_rows":int(frame.execution_label_end_utc.notna().sum()),"base_score_rows":int(frame[BASE].notna().sum()),"residual_score_rows":int(frame[RESIDUAL].notna().sum()),"alpha_target_rows":int(frame[ALPHA].notna().sum()),"execution_target_rows":int(frame[TARGET].notna().sum())}
        for name,frame in (("historical",historical_ledger),("forward",forward_ledger))
    ])
    if history.__ts__.max()>=pd.Timestamp("2026-01-01",tz="UTC") or current.__ts__.min()<pd.Timestamp("2026-01-01",tz="UTC"): raise RegimeOOFStackError("historical/current split is invalid")
    all_oof=[];all_audit=[];forward=[];summary=[];periods=[];sides=[];recall=[];calibration=[];start=pd.Timestamp(oof_start)
    for number,arm in enumerate(ARMS):
        oof,audit=_oof(history,arm,start=start,min_train=min_train_rows);mapper=_mapper(oof.raw_score.to_numpy(float),oof[TARGET].to_numpy(float));oof["mapped_score"]=mapper(oof.raw_score.to_numpy(float));all_oof.append(oof);all_audit.append(audit)
        age=(current.__ts__.min()-pd.to_datetime(oof.execution_label_end_utc,utc=True).max()).days
        if age>max_map_age_days: raise RegimeOOFStackError(f"{arm.name} causal map is stale by {age}d; maximum is {max_map_age_days}d")
        parts=[]
        for side,local in current.groupby("side_name",observed=True):
            fit=history.loc[history.side_name.eq(side)];raw,model=_predict(fit,local,arm,12000+number*17+(side=="short"));parts.append(local.assign(arm=arm.name,raw_score=raw,mapped_score=mapper(raw),map_source_last_label_end_utc=oof.execution_label_end_utc.max(),map_age_days=age))
        result=pd.concat(parts,ignore_index=True);row,per,side,rec,cal=_evaluate(result,arm);row["map_age_days"]=age;forward.append(result);summary.append(row);periods.append(per);sides.append(side);recall.append(rec);calibration.append(cal)
    temporary=Path(tempfile.mkdtemp(dir=output.parent,prefix=f".{output.name}."))
    try:
        pd.concat(all_oof,ignore_index=True).to_parquet(temporary/"historical_oof_scores.parquet",index=False);pd.concat(all_audit,ignore_index=True).to_parquet(temporary/"oof_fit_audit.parquet",index=False);pd.concat(forward,ignore_index=True).to_parquet(temporary/"frozen_2026_candidate_scores.parquet",index=False);pd.DataFrame(summary).to_csv(temporary/"metrics_summary.csv",index=False);pd.concat(periods,ignore_index=True).to_parquet(temporary/"period_metrics.parquet",index=False);pd.concat(sides,ignore_index=True).to_parquet(temporary/"side_metrics.parquet",index=False);pd.concat(recall,ignore_index=True).to_csv(temporary/"recall.csv",index=False);pd.concat(calibration,ignore_index=True).to_parquet(temporary/"calibration_deciles.parquet",index=False);intersection_coverage.to_csv(temporary/"context_intersection_coverage.csv",index=False);input_score_coverage.to_csv(temporary/"input_score_coverage.csv",index=False)
        contract={"sidecar_manifest_schema":SIDECAR_SCHEMA,"candidate_cadence":"1h","labels":"1m only nested in existing exact 12h labels","historical_score_aliases":"2023-2024 canonical score columns plus complementary 2025 base_oof_score/residual_expected_ev are conflict-checked and coalesced; execution_label_available_at supplies the complete 13h label-resolution boundary","historical_fit_selection_calibration":"available strictly OOF pre-2026 candidate rows only; expanding 3-month OOF blocks; label end strictly before fold","forward":"2026 only; no label used in model fit or map","context":"continuous semantic BOCPD regime and LGBM/BOCPD transition probabilities only; state IDs/raw posterior axes forbidden","identical_row_universe":"all arms use one frozen context-available candidate intersection; explicit pre-OOF warm-up rows are reported and excluded, never imputed","mapping":"monotone increasing OOF isotonic because every upstream score is higher-is-better; automatic direction is forbidden","selection":"one pooled global top10 after arm-local frozen historical OOF isotonic EV map; exact mapped ties use higher raw score then candidate_id only as deterministic secondary keys; period metrics decompose fixed membership","arms":[arm.__dict__ for arm in ARMS]};_write_json(temporary/"contract.json",contract)
        files=[p for p in temporary.iterdir() if p.is_file()];final={"schema":SCHEMA,"status":"SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION","promotion_eligible":False,"source_soft_sidecar_manifest_sha256":sha(Path(sidecar_manifest)),"inputs":{str(Path(p).resolve()):sha(Path(p)) for p in (sidecar_manifest,historical_scores,current_scores,regime_path,transition_path)},"contract":contract,"outputs_sha256":{p.name:sha(p) for p in files}};_write_json(temporary/"manifest.json",final);(temporary/"manifest.sha256").write_text(f"{sha(temporary/'manifest.json')}  manifest.json\n");os.replace(temporary,output);return output
    except Exception: shutil.rmtree(temporary,ignore_errors=True);raise


def parse_args(argv:Sequence[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--sidecar-manifest",type=Path,required=True);p.add_argument("--historical-scores",type=Path,required=True);p.add_argument("--current-scores",type=Path,required=True);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--oof-start",default="2023-01-01T00:00:00Z");p.add_argument("--min-train-rows",type=int,default=12000);p.add_argument("--max-map-age-days",type=int,default=90);return p.parse_args(argv)
if __name__=="__main__": print(run(**vars(parse_args())))
