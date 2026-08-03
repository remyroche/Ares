#!/usr/bin/env python3
"""All-era frozen-unsupervised economic validation with explicit fail-closed arms.

The unsupervised representations are selected/fitted on the 2022-08--2025
hourly regime panel.  Economic conversion is fitted and chronologically OOF
mapped on the separately documented pre-2026 candidate ledgers, then frozen
and assessed once on the semantically identical May--July 2026 candidate
intersection.  No 2026 realised outcome is used in any fit or EV map.

The diagonal GMM and failure-first arms deliberately emit availability rows
rather than being silently substituted: their currently sealed historical
sidecars cannot be joined to the same historical residual-score/economics
contract.  This is stricter than the older May--July-only common OOF runner.
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
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, validate_candidate_identity  # noqa: E402
from scripts.run_strict_forward_dae_gmm_regime_challenger import DenoisingAE, density_input  # noqa: E402
from scripts.run_strict_forward_sticky_fullcov_regime_challenger import causal_filter  # noqa: E402

SCHEMA = "unsupervised_economic_all_era_strict_v1"
OUT = ROOT / "data_perp/artifacts/unsupervised_economic_all_era_strict_20260730_v1"
HIST_2022_24 = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4/oof_scores.parquet"
HIST_2025 = ROOT / "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1/all_score_waterfall.parquet"
CURRENT_2026 = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1/allscore_waterfall.parquet"
PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
STICKY = ROOT / "data_perp/artifacts/strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1"
DAE = ROOT / "data_perp/artifacts/strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1"
DIAGONAL = ROOT / "data_perp/artifacts/strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3"
FAILURE = ROOT / "data_perp/artifacts/failure_first_detector_current_transfer_20260726_v6/candidate_overlay.parquet"

START, CUT = pd.Timestamp("2022-08-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
TARGET, GROSS, COST, SCORE, ALPHA = "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "score_residual_expected_ev", "__first_touch_target_soft__"
TOP = .10
ARMS = {
    "baseline": [SCORE],
    "sticky_fullcov_gmm_geometry": [SCORE, "sticky_entropy", "sticky_margin", "sticky_ood_score", "sticky_is_ood"],
    "dae_to_gmm_geometry": [SCORE, "dae_entropy", "dae_margin", "dae_density_ood_score", "dae_reconstruction_error", "dae_is_ood"],
}


def status(message: str) -> None:
    if os.environ.get("UNSUPERVISED_ECONOMIC_PROGRESS") == "1":
        print(message, flush=True)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def _numeric(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    z = test.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median().fillna(0.0)
    return x.fillna(med).to_numpy(np.float32), z.fillna(med).to_numpy(np.float32)


def _score_model(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], seed: int) -> np.ndarray:
    x, z = _numeric(train, test, fields)
    model = lgb.LGBMRegressor(
        n_estimators=140, learning_rate=.04, num_leaves=15, min_child_samples=220,
        subsample=.85, colsample_bytree=.9, reg_lambda=4., random_state=seed,
        n_jobs=4, verbosity=-1,
    ).fit(x, train[TARGET].to_numpy(float))
    return np.asarray(model.predict(z), float)


def _mapper(scores: np.ndarray, target: np.ndarray):
    valid = np.isfinite(scores) & np.isfinite(target)
    scores, target = scores[valid], target[valid]
    if len(scores) < 8 or np.unique(scores).size < 2:
        value = float(target.mean()) if len(target) else 0.0
        return lambda values: np.full(len(values), value)
    model = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(scores, target)
    return lambda values: np.asarray(model.predict(np.asarray(values, float)), float)


def _transform(bundle: dict[str, Any], panel: pd.DataFrame) -> np.ndarray:
    raw = bundle["imputer"].transform(panel[bundle["features"]])
    return bundle["scaler"].transform(np.clip(raw, bundle["lower"], bundle["upper"]))


def _representation_batched(net: DenoisingAE, values: np.ndarray, batch_rows: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Bound peak Torch allocation when replaying a frozen DAE on history."""
    latents: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_rows):
            block = values[start:start + batch_rows].astype(np.float32, copy=False)
            tensor = torch.from_numpy(block)
            latent = net.encoder(tensor).cpu().numpy()
            decoded = net(tensor).cpu().numpy()
            latents.append(latent)
            errors.append(np.mean(np.square(block - decoded), axis=1))
    return np.concatenate(latents).astype(np.float64), np.concatenate(errors).astype(np.float64)


def _geometry_historical(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply sealed 2022--25 unsupervised identities to historical hours only.

    This is representation-in-sample but outcome-free.  It is admissible as a
    historical feature for the subsequent chronological *economic* OOF; it is
    never called a regime OOF sidecar.
    """
    historic = panel.loc[(panel.source_utc >= START) & (panel.source_utc < CUT)].copy()
    segments = historic.calendar_segment_id.astype(str).to_numpy()
    sticky = joblib.load(STICKY / "frozen_regime_model.joblib")
    values = _transform(sticky, historic)
    emissions = sticky["gmm"].predict_proba(values)
    filtered, _, _ = causal_filter(emissions, segments, sticky["transition"])
    ordered = np.sort(filtered, axis=1)
    out = historic.loc[:, ["source_utc"]].copy()
    out["sticky_entropy"] = -(np.clip(filtered, 1e-12, 1) * np.log(np.clip(filtered, 1e-12, 1))).sum(axis=1) / np.log(filtered.shape[1])
    out["sticky_margin"] = ordered[:, -1] - ordered[:, -2]
    out["sticky_ood_score"] = -sticky["gmm"].score_samples(values)
    out["sticky_is_ood"] = out["sticky_ood_score"].gt(float(sticky["ood_threshold"])).astype(float)

    dae_bundle = joblib.load(DAE / "frozen_dae_gmm_transform_and_density.joblib")
    checkpoint = torch.load(DAE / "frozen_dae_state_dict.pt", map_location="cpu", weights_only=True)
    net = DenoisingAE(int(checkpoint["inputs"]), int(checkpoint["bottleneck"]))
    net.load_state_dict(checkpoint["state_dict"]); net.eval()
    dae_values = _transform(dae_bundle, historic)
    latent, error = _representation_batched(net, dae_values)
    # The density scale is derived from the same outcome-free 2022--25 fit
    # population that trained the frozen DAE/GMM; no current rows are used.
    density = density_input(latent, error, error)
    emissions = dae_bundle["gmm"].predict_proba(density)
    filtered, _, _ = causal_filter(emissions, segments, dae_bundle["transition"])
    ordered = np.sort(filtered, axis=1)
    out["dae_entropy"] = -(np.clip(filtered, 1e-12, 1) * np.log(np.clip(filtered, 1e-12, 1))).sum(axis=1) / np.log(filtered.shape[1])
    out["dae_margin"] = ordered[:, -1] - ordered[:, -2]
    out["dae_density_ood_score"] = -dae_bundle["gmm"].score_samples(density)
    out["dae_reconstruction_error"] = error
    out["dae_is_ood"] = ((out["dae_density_ood_score"] > float(dae_bundle["density_ood_threshold"])) | (out["dae_reconstruction_error"] > float(dae_bundle["reconstruction_ood_threshold"]))).astype(float)
    return out


def _load_historical() -> pd.DataFrame:
    old_columns = [*IDENTITY_COLUMNS, "stack_lineage", TARGET, GROSS, COST, SCORE]
    old = pd.read_parquet(HIST_2022_24, columns=old_columns)
    # The inverse-PI 2022 segment is a documented different candidate
    # population, so do not pretend it shares the frozen-PF contract.
    old = old.loc[old.stack_lineage.eq("frozen_pf_2022aug_2024")].copy()
    old["execution_label_end_utc"] = pd.to_datetime(old["__ts__"], utc=True) + pd.Timedelta(hours=12)
    columns = [*IDENTITY_COLUMNS, "execution_label_end_utc", TARGET, GROSS, COST, SCORE]
    recent = pd.read_parquet(HIST_2025, columns=columns)
    result = pd.concat([old.loc[:, columns], recent.loc[:, columns]], ignore_index=True)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["execution_label_end_utc"] = pd.to_datetime(result["execution_label_end_utc"], utc=True)
    if (result.execution_label_end_utc <= result.__ts__).any() or not (result.__ts__.dt.minute.eq(0)).all():
        raise RegimeOOFStackError("historical candidate timing/cadence contract failed")
    return validate_candidate_identity(result).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_current() -> pd.DataFrame:
    columns = [*IDENTITY_COLUMNS, "execution_label_end_utc", TARGET, GROSS, COST, SCORE, ALPHA]
    current = validate_candidate_identity(pd.read_parquet(CURRENT_2026, columns=columns))
    failure = validate_candidate_identity(pd.read_parquet(FAILURE, columns=list(IDENTITY_COLUMNS)))
    current = current.merge(failure, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    current = current.loc[:, [*IDENTITY_COLUMNS, "execution_label_end_utc", TARGET, GROSS, COST, SCORE, ALPHA]].copy()
    current["execution_label_end_utc"] = pd.to_datetime(current["execution_label_end_utc"], utc=True)
    current["__ts__"] = pd.to_datetime(current["__ts__"], utc=True)
    return validate_candidate_identity(current).sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _forward_geometry_current(current: pd.DataFrame) -> pd.DataFrame:
    def read(path: Path, rename: dict[str, str], fields: list[str]) -> pd.DataFrame:
        q = pd.read_parquet(path, columns=["source_utc", "regime_available_utc", *fields]).rename(columns={"source_utc": "__ts__", **rename})
        q["__ts__"] = pd.to_datetime(q["__ts__"], utc=True); q["regime_available_utc"] = pd.to_datetime(q["regime_available_utc"], utc=True)
        if q.duplicated("__ts__").any() or q.regime_available_utc.gt(q.__ts__).any():
            raise RegimeOOFStackError(f"invalid strict forward regime sidecar: {path}")
        return q.drop(columns="regime_available_utc")
    sticky = read(STICKY / "regime_only_forward_2026_sidecar.parquet", {"regime_entropy":"sticky_entropy", "regime_margin":"sticky_margin", "regime_ood_score":"sticky_ood_score", "regime_is_ood":"sticky_is_ood"}, ["regime_entropy", "regime_margin", "regime_ood_score", "regime_is_ood"])
    dae = read(DAE / "regime_only_forward_2026_sidecar.parquet", {"regime_entropy":"dae_entropy", "regime_margin":"dae_margin", "regime_density_ood_score":"dae_density_ood_score", "regime_reconstruction_error":"dae_reconstruction_error", "regime_is_ood":"dae_is_ood"}, ["regime_entropy", "regime_margin", "regime_density_ood_score", "regime_reconstruction_error", "regime_is_ood"])
    out = current.merge(sticky, on="__ts__", how="inner", validate="many_to_one").merge(dae, on="__ts__", how="inner", validate="many_to_one")
    return validate_candidate_identity(out)


def _historical_oof(history: pd.DataFrame, arm: str, fields: list[str]) -> pd.DataFrame:
    # Six-month blocks provide several large, chronological OOF references
    # without repeatedly refitting nearly-identical 400k-row models.
    start = history.__ts__.min().normalize() + pd.DateOffset(months=12)
    starts = pd.date_range(start, history.__ts__.max().normalize() + pd.Timedelta(days=1), freq="6MS", tz="UTC")
    rows: list[pd.DataFrame] = []
    for number, block in enumerate(starts):
        end = block + pd.DateOffset(months=6)
        test = history.loc[(history.__ts__ >= block) & (history.__ts__ < end)].copy()
        train = history.loc[history.execution_label_end_utc < block].copy()
        if test.empty or len(train) < 30_000:
            continue
        if train.execution_label_end_utc.max() >= block:
            raise RegimeOOFStackError("historical OOF training label-resolution gate failed")
        for side, local in test.groupby("side_name", observed=True):
            fit = train.loc[train.side_name.eq(side)]
            if len(fit) < 12_000:
                continue
            raw = _score_model(fit, local, fields, 7301 + number * 19 + (side == "short"))
            rows.append(local.loc[:, [*IDENTITY_COLUMNS, "execution_label_end_utc", TARGET]].assign(arm=arm, fold_start_utc=block, train_label_end_max=train.execution_label_end_utc.max(), raw_score=raw))
    if not rows:
        raise RegimeOOFStackError(f"no historical OOF support for {arm}")
    result = pd.concat(rows, ignore_index=True)
    return validate_candidate_identity(result)


def _fixed_global_metrics(frame: pd.DataFrame, arm: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = frame.sort_values(["mapped_score", "candidate_id"], ascending=[False, True], kind="stable")
    chosen = order.head(max(1, math.ceil(len(frame) * TOP))).copy()
    selected_ids = set(chosen.candidate_id)
    frame = frame.copy(); frame["selected_global_top10"] = frame.candidate_id.isin(selected_ids)
    def rank(left: pd.Series, right: pd.Series) -> float:
        ok = left.notna() & right.notna()
        return float(left.loc[ok].rank().corr(right.loc[ok].rank())) if ok.sum() >= 3 else float("nan")
    records=[]
    for kind, keys in (("week", frame.__ts__.dt.strftime("%G-W%V")), ("month", frame.__ts__.dt.strftime("%Y-%m"))):
        for period, local in frame.groupby(keys, sort=True, observed=True):
            selected = local.loc[local.selected_global_top10]
            records.append({"arm":arm,"period_type":kind,"period":period,"candidate_rows":len(local),"global_selected_rows":len(selected),"alpha_rank_ic":rank(local.mapped_score,local[ALPHA]),"execution_net_rank_ic":rank(local.mapped_score,local[TARGET]),"mean_net_ev":selected[TARGET].mean(),"mean_gross_ev":selected[GROSS].mean(),"mean_cost":selected[COST].mean(),"hit_rate":selected[TARGET].gt(0).mean()})
    periods=pd.DataFrame(records)
    summary={"arm":arm,"selection_basis":"one_pooled_global_top10_after_arm_local_frozen_historical_oof_ev_map","candidate_rows":len(frame),"top10_support":len(chosen),"alpha_rank_ic":rank(frame.mapped_score,frame[ALPHA]),"execution_net_rank_ic":rank(frame.mapped_score,frame[TARGET]),"top10_mean_net_ev":chosen[TARGET].mean(),"top10_mean_gross_ev":chosen[GROSS].mean(),"top10_mean_cost":chosen[COST].mean(),"top10_hit_rate":chosen[TARGET].gt(0).mean()}
    for kind in ("week","month"):
        local=periods.loc[periods.period_type.eq(kind)]
        for prefix,column in (("ic","alpha_rank_ic"),("net_ev","mean_net_ev")):
            summary[f"{kind}_{prefix}_q10"]=local[column].quantile(.10);summary[f"{kind}_{prefix}_q50"]=local[column].quantile(.50)
    sides=[]
    for side, local in frame.groupby("side_name",observed=True):
        selected=local.loc[local.selected_global_top10];assets=selected.__symbol__.value_counts(normalize=True)
        sides.append({"arm":arm,"side_name":side,"candidate_rows":len(local),"global_selected_rows":len(selected),"execution_net_rank_ic":rank(local.mapped_score,local[TARGET]),"global_top10_net_ev":selected[TARGET].mean(),"global_top10_hit_rate":selected[TARGET].gt(0).mean(),"selected_asset_count":selected.__symbol__.nunique(),"selected_largest_asset_share":assets.iloc[0] if len(assets) else np.nan,"selected_asset_hhi":(assets**2).sum() if len(assets) else np.nan})
    assets=chosen.__symbol__.value_counts(normalize=True)
    concentration=pd.DataFrame([{"arm":arm,"selected_asset_count":chosen.__symbol__.nunique(),"selected_largest_asset_share":assets.iloc[0],"selected_asset_hhi":(assets**2).sum(),"long_share":chosen.side_name.eq("long").mean()}])
    return summary, periods, pd.DataFrame(sides), concentration


def run(output: Path = OUT) -> Path:
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    # The canonical hourly panel is deliberately wide.  This economic
    # validation needs only the two already-frozen representation contracts,
    # so materialise their union rather than copying the entire research plane.
    sticky_contract = joblib.load(STICKY / "frozen_regime_model.joblib")
    dae_contract = joblib.load(DAE / "frozen_dae_gmm_transform_and_density.joblib")
    hourly_columns = list(dict.fromkeys(["source_utc", "calendar_segment_id", *sticky_contract["features"], *dae_contract["features"]]))
    status("loading frozen hourly representation fields")
    hourly=pd.read_parquet(PANEL, columns=hourly_columns)
    hourly["source_utc"]=pd.to_datetime(hourly.source_utc,utc=True)
    historic=_load_historical(); current=_load_current();status(f"loaded historical={len(historic)} current={len(current)}")
    geometry=_geometry_historical(hourly);status("replayed frozen historical geometry")
    historic=historic.merge(geometry,left_on="__ts__",right_on="source_utc",how="inner",validate="many_to_one").drop(columns="source_utc")
    current=_forward_geometry_current(current)
    if not (historic.__ts__.dt.minute.eq(0).all() and current.__ts__.dt.minute.eq(0).all()): raise RegimeOOFStackError("non-hourly candidate row")
    availability=pd.DataFrame([
        {"arm":"baseline","status":"available_semantically_identical_residual_ev_and_exact_12h_economics","historical_rows":len(historic),"current_rows":len(current),"reason":"same residual-expected-EV score and explicit gross/cost/net contract"},
        {"arm":"sticky_fullcov_gmm_geometry","status":"available_frozen_2022_2025_hourly_representation","historical_rows":len(historic),"current_rows":len(current),"reason":"serialized frozen transform applies the same geometry fields to both cohorts"},
        {"arm":"dae_to_gmm_geometry","status":"available_frozen_2022_2025_hourly_representation","historical_rows":len(historic),"current_rows":len(current),"reason":"serialized frozen DAE/GMM transform applies the same geometry fields to both cohorts"},
        {"arm":"diagonal_gmm_geometry","status":"fail_closed_missing_serialized_historical_transform","historical_rows":0,"current_rows":len(current),"reason":"sealed diagonal artifact has 2026 sidecar but no immutable historical transform; no substitute allowed"},
        {"arm":"failure_first_context","status":"fail_closed_no_semantically_identical_pre2026_joint_candidate_score_overlay","historical_rows":0,"current_rows":len(current),"reason":"historical failure OOF overlay is Nov-2025-only and has no exact intersection with the residual-EV economics training ledger"},
    ])
    oof_parts=[]; forward_parts=[]; summaries=[];period_parts=[];side_parts=[];conc=[]
    for number,(arm,fields) in enumerate(ARMS.items()):
        status(f"historical OOF {arm}")
        oof=_historical_oof(historic,arm,fields); mapper=_mapper(oof.raw_score.to_numpy(float),oof[TARGET].to_numpy(float));oof["mapped_score"]=mapper(oof.raw_score.to_numpy(float));oof_parts.append(oof)
        status(f"frozen forward score {arm}")
        scored=[]
        for side,local in current.groupby("side_name",observed=True):
            fit=historic.loc[historic.side_name.eq(side)];raw=_score_model(fit,local,fields,9101+number*13+(side=="short"));scored.append(local.assign(arm=arm,raw_score=raw,mapped_score=mapper(raw)))
        forward=pd.concat(scored,ignore_index=True); summary,periods,sides,cx=_fixed_global_metrics(forward,arm);forward_parts.append(forward);summaries.append(summary);period_parts.append(periods);side_parts.append(sides);conc.append(cx)
    status("writing sealed artifact")
    tmp=Path(tempfile.mkdtemp(dir=output.parent,prefix=f".{output.name}."))
    try:
        pd.concat(oof_parts,ignore_index=True).to_parquet(tmp/"historical_economic_oof_scores.parquet",index=False)
        pd.concat(forward_parts,ignore_index=True).to_parquet(tmp/"frozen_2026_candidate_scores.parquet",index=False)
        pd.DataFrame(summaries).to_csv(tmp/"metrics_summary.csv",index=False);pd.concat(period_parts,ignore_index=True).to_parquet(tmp/"period_metrics.parquet",index=False);pd.concat(side_parts,ignore_index=True).to_parquet(tmp/"side_metrics.parquet",index=False);pd.concat(conc,ignore_index=True).to_csv(tmp/"concentration_metrics.csv",index=False);availability.to_csv(tmp/"arm_availability.csv",index=False)
        contract={"candidate_cadence":"1h","replay_cadence":"1m nested only in exact 12h labels","representation_fit_selection":"2022-08 through 2025 only; no 2026 row used","economic_fit_map":"pre-2026 candidates only; chronological OOF raw scores/outcomes only","forward_assessment":"frozen 2026 score/model/map; no 2026 label in fit or map","selection":"one pooled global top10 after each arm-local frozen historical OOF EV map; period tables decompose that fixed membership and never rerank","historical_lineages":"frozen-PF 2022-08..2024 plus strict Mar-Apr 2025; inverse-PI 2022 is excluded","arms":ARMS}
        _atomic_json(tmp/"feature_and_lineage_contract.json",contract)
        files=[p for p in tmp.iterdir() if p.is_file()]
        manifest={"schema":SCHEMA,"status":"SEALED_STRICT_FORWARD_ECONOMIC_DIAGNOSTIC_NON_PROMOTION","promotion_eligible":False,"portfolio_replay":False,"inputs":{str(p):sha(p) for p in (HIST_2022_24,HIST_2025,CURRENT_2026,PANEL,STICKY/"frozen_regime_model.joblib",DAE/"frozen_dae_gmm_transform_and_density.joblib",DAE/"frozen_dae_state_dict.pt",FAILURE,DIAGONAL/"regime_only_forward_2026_sidecar.parquet")},"outputs_sha256":{p.name:sha(p) for p in files},"availability":{"available":availability.loc[availability.status.str.startswith("available"),"arm"].tolist(),"fail_closed":availability.loc[availability.status.str.startswith("fail_closed"),"arm"].tolist()},"contract":contract}
        _atomic_json(tmp/"manifest.json",manifest);(tmp/"manifest.sha256").write_text(f"{sha(tmp/'manifest.json')}  manifest.json\n");os.replace(tmp,output);return output
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True);raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=OUT);return parser.parse_args(argv)


if __name__ == "__main__": print(run(**vars(parse_args())))
