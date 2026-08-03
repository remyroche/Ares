#!/usr/bin/env python3
"""Causal baseline EV-map refresh using sealed July common-30 blocked OOF.

This is deliberately baseline-only.  It fits four predeclared maps: the
historical cutoff and historical+July common-30 cutoff, each with ordinary and
strict-rank-preserving isotonic output.  The frozen 2026 population is only
assessed, never used to select a map or fit a parameter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
STACK = ROOT / "data_perp/artifacts/final_identical_row_regime_stack_gam_ablation_20260730_v3"
JULY = ROOT / "data_perp/artifacts/july2025_common30_final_base_residual_oof_bridge_20260730_v1"
OUT = ROOT / "data_perp/artifacts/july_common30_baseline_map_refresh_20260730_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TARGET, GROSS, COST, ALPHA = "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "__first_touch_target_soft__"
TOP = .10


class RefreshError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _sealed_stack(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path, marker = root / "manifest.json", root / "manifest.sha256"
    if not manifest_path.is_file() or not marker.is_file() or marker.read_text().split(maxsplit=1)[0] != sha(manifest_path):
        raise RefreshError("frozen v3 stack manifest is not sealed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "final_identical_row_regime_stack_gam_ablation_v3" or manifest.get("status") != "SEALED_STRICT_FORWARD_IDENTICAL_ROW_ABLATION_NON_PROMOTION":
        raise RefreshError("requires the corrected sealed v3 stack")
    history_path, forward_path = root / "historical_oof_scores.parquet", root / "frozen_2026_candidate_scores.parquet"
    for path in (history_path, forward_path):
        if manifest.get("outputs_sha256", {}).get(path.name) != sha(path):
            raise RefreshError(f"v3 input checksum mismatch: {path}")
    history = pd.read_parquet(history_path, filters=[("arm", "==", "baseline")])
    forward = pd.read_parquet(forward_path, filters=[("arm", "==", "baseline")])
    for name, frame in (("history", history), ("forward", forward)):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["execution_label_end_utc"] = pd.to_datetime(frame["execution_label_end_utc"], utc=True, errors="raise")
        if frame.duplicated(IDENTITY).any() or not (frame["__ts__"].astype("int64") % pd.Timedelta(hours=1).value == 0).all():
            raise RefreshError(f"{name} is not an exact unique hourly candidate ledger")
    if not history["__ts__"].lt(pd.Timestamp("2026-01-01", tz="UTC")).all() or not forward["__ts__"].ge(pd.Timestamp("2026-01-01", tz="UTC")).all():
        raise RefreshError("v3 split is not strictly pre-2026 versus 2026")
    return history, forward, manifest


def _sealed_july(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract_path, marker = root / "bridge_contract.json", root / "manifest.sha256"
    if not contract_path.is_file() or not marker.is_file():
        raise RefreshError("July bridge seal is missing")
    contract = json.loads(contract_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if marker.read_text().split(maxsplit=1)[0] != sha(root / "manifest.json") or contract.get("status") != "SEALED_COMMON30_BLOCKED_OOF_BRIDGE_NON_PROMOTION":
        raise RefreshError("July bridge is not sealed")
    path = root / "oof_predictions.parquet"
    expected = contract.get("outputs", {}).get(path.name) or manifest.get("outputs_sha256", {}).get(path.name)
    if not expected or expected != sha(path):
        raise RefreshError("July bridge score checksum mismatch")
    cols = [*IDENTITY, "score_residual_expected_ev", TARGET, "execution_label_end_utc", "residual_is_oof"]
    july = pd.read_parquet(path, columns=cols)
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True, errors="raise")
    july["execution_label_end_utc"] = pd.to_datetime(july["execution_label_end_utc"], utc=True, errors="raise")
    if len(july) != 44_640 or july.duplicated(IDENTITY).any() or not july.residual_is_oof.all():
        raise RefreshError("July bridge does not prove exact both-side residual OOF coverage")
    if not july["__ts__"].between(pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-07-31 23:00", tz="UTC")).all():
        raise RefreshError("July bridge dates are not exact")
    if not (july["__ts__"].astype("int64") % pd.Timedelta(hours=1).value == 0).all() or not july.execution_label_end_utc.gt(july["__ts__"]).all():
        raise RefreshError("July bridge cadence or label endpoint is invalid")
    july = july.rename(columns={"score_residual_expected_ev": "raw_score"})
    return july, contract


def _fit(history: pd.DataFrame):
    x, y = history.raw_score.to_numpy(float), history[TARGET].to_numpy(float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8 or np.unique(x[good]).size < 2:
        raise RefreshError("insufficient score support for causal map")
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[good], y[good])


def _strict_rank(mapped: np.ndarray, raw: np.ndarray) -> np.ndarray:
    rank = pd.Series(raw).rank(method="first", pct=True).to_numpy(float)
    scale = max(float(np.nanmax(np.abs(mapped))), 1e-4)
    return mapped + rank * scale * 1e-12


def _select(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["mapped_score", "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable")
    chosen = set(ordered.head(max(1, math.ceil(len(ordered) * TOP))).candidate_id)
    out = frame.copy(); out["selected_global_top10"] = out.candidate_id.isin(chosen)
    return out


def _rank(a: pd.Series, b: pd.Series) -> float:
    return float(a.corr(b, method="spearman"))


def _ties(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame.mapped_score.value_counts(dropna=False)
    cutoff = frame.sort_values(["mapped_score", "raw_score", "candidate_id"], ascending=[False, False, True], kind="stable").iloc[math.ceil(len(frame)*TOP)-1].mapped_score
    return {"mapped_unique_values": int(frame.mapped_score.nunique()), "mapped_tie_mass": float(counts.max()/len(frame)), "cutoff_tie_rows": int(frame.mapped_score.eq(cutoff).sum()), "cutoff_tie_share": float(frame.mapped_score.eq(cutoff).mean())}


def _evaluate(frame: pd.DataFrame, name: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _select(frame); pick = frame.loc[frame.selected_global_top10]
    rows=[]
    for kind, key in (("week", frame.__ts__.dt.strftime("%G-W%V")), ("month", frame.__ts__.dt.strftime("%Y-%m"))):
        for period, local in frame.groupby(key, observed=True, sort=True):
            chosen=local.loc[local.selected_global_top10]
            rows.append({"map":name,"period_type":kind,"period":period,"candidate_rows":len(local),"global_selected_rows":len(chosen),"mean_net_ev":chosen[TARGET].mean(),"mean_gross_ev":chosen[GROSS].mean(),"mean_cost":chosen[COST].mean(),"hit_rate":chosen[TARGET].gt(0).mean()})
    periods=pd.DataFrame(rows)
    summary={"map":name,"candidate_rows":len(frame),"top10_rows":len(pick),"alpha_rank_ic":_rank(frame.mapped_score,frame[ALPHA]),"execution_rank_ic":_rank(frame.mapped_score,frame[TARGET]),"raw_score_net_rank_ic":_rank(frame.raw_score,frame[TARGET]),"top10_net_ev":pick[TARGET].mean(),"top10_gross_ev":pick[GROSS].mean(),"top10_cost":pick[COST].mean(),"top10_hit_rate":pick[TARGET].gt(0).mean(),**_ties(frame)}
    for kind in ("week","month"):
        x=periods.loc[periods.period_type.eq(kind)]
        summary[f"{kind}_net_ev_q10"]=x.mean_net_ev.quantile(.10); summary[f"{kind}_net_ev_q50"]=x.mean_net_ev.quantile(.50)
        latest=x.sort_values("period").iloc[-1]; worst=x.loc[x.mean_net_ev.idxmin()]
        summary[f"latest_{kind}"]=latest.period; summary[f"latest_{kind}_net_ev"]=latest.mean_net_ev
        summary[f"worst_{kind}"]=worst.period; summary[f"worst_{kind}_net_ev"]=worst.mean_net_ev
    sides=[]
    for side, x in frame.groupby("side_name", observed=True, sort=True):
        s=x.loc[x.selected_global_top10]
        sides.append({"map":name,"side_name":side,"candidate_rows":len(x),"global_selected_rows":len(s),"execution_rank_ic":_rank(x.mapped_score,x[TARGET]),"top10_net_ev":s[TARGET].mean(),"top10_gross_ev":s[GROSS].mean(),"top10_cost":s[COST].mean(),"top10_hit_rate":s[TARGET].gt(0).mean()})
    calibration=frame[["mapped_score",TARGET]].copy(); calibration["decile"]=pd.qcut(calibration.mapped_score.rank(method="first"),10,labels=False)
    calibration=calibration.groupby("decile",observed=True).agg(candidate_rows=(TARGET,"size"),mean_mapped_ev=("mapped_score","mean"),mean_net_ev=(TARGET,"mean")).reset_index(); calibration["map"]=name; calibration["signed_error"]=calibration.mean_mapped_ev-calibration.mean_net_ev
    summary["calibration_mae_decile"]=calibration.signed_error.abs().mean()
    return summary,periods,pd.DataFrame(sides),calibration


def run(*, stack: Path = STACK, july_root: Path = JULY, output: Path = OUT) -> Path:
    output=Path(output)
    if output.exists(): raise RefreshError(f"immutable output exists: {output}")
    old, forward, stack_manifest = _sealed_stack(Path(stack)); july, july_contract = _sealed_july(Path(july_root))
    old_fit=old[[*IDENTITY,"raw_score",TARGET,"execution_label_end_utc"]].copy()
    july_fit=july[[*IDENTITY,"raw_score",TARGET,"execution_label_end_utc"]].copy()
    histories={"old_cutoff":old_fit,"july_refreshed_common30":pd.concat([old_fit,july_fit],ignore_index=True)}
    results=[]; periods=[]; sides=[]; calibrations=[]; selected=[]; audit=[]
    for base_name, history in histories.items():
        model=_fit(history)
        for rank_mode in (False,True):
            name=f"{base_name}_{'rank_preserving' if rank_mode else 'isotonic'}"
            scored=forward.copy(); mapped=model.predict(scored.raw_score.to_numpy(float)); scored["mapped_score"]=_strict_rank(mapped,scored.raw_score.to_numpy(float)) if rank_mode else mapped
            summary,per,side,cal=_evaluate(scored,name); results.append(summary);periods.append(per);sides.append(side);calibrations.append(cal)
            scored=_select(scored); selected.append(scored.loc[scored.selected_global_top10,[*IDENTITY,"raw_score","mapped_score",TARGET,GROSS,COST,"execution_label_end_utc","selected_global_top10"]].assign(map=name))
            audit.append({"map":name,"fit_rows":int(len(history)),"fit_label_end_max":history.execution_label_end_utc.max(),"fit_is_strict_pre2026":bool(history.__ts__.lt(pd.Timestamp("2026-01-01",tz="UTC")).all()),"july_common30_rows":int(len(july_fit) if base_name.startswith("july_") else 0),"rank_preserving":rank_mode,"no_2026_fit_tuning_or_selection":True})
    stage=Path(tempfile.mkdtemp(dir=output.parent,prefix=f".{output.name}."))
    try:
        pd.DataFrame(results).to_csv(stage/"metrics_summary.csv",index=False); pd.concat(periods,ignore_index=True).to_parquet(stage/"period_metrics.parquet",index=False); pd.concat(sides,ignore_index=True).to_parquet(stage/"side_metrics.parquet",index=False);pd.concat(calibrations,ignore_index=True).to_parquet(stage/"calibration_deciles.parquet",index=False);pd.concat(selected,ignore_index=True).to_parquet(stage/"frozen_2026_selected_scores.parquet",index=False);write_json(stage/"mapping_fit_audit.json",audit)
        contract={"sample_cadence":"1h","exact_replay_bar_cadence":"1m_labels_only","fit":"baseline residual score isotonic map only; frozen model/context arms untouched","forward_assessment":"identical frozen 127777 2026 hourly candidates; global pooled top10","maps":"old cutoff and old+sealed-July-common30, ordinary/rank-preserving isotonic; fixed before reading 2026 outcomes","no_2026_fit_tuning_or_selection":True,"scope_limitation":"July adds only a frozen 30-asset common-universe OOF cohort; it is not identical to the wider v3 candidate population and cannot promote or replace it","source_stack_manifest_sha256":sha(Path(stack)/"manifest.json"),"source_july_manifest_sha256":sha(Path(july_root)/"manifest.json"),"old_label_end_max":old_fit.execution_label_end_utc.max(),"july_label_end_max":july_fit.execution_label_end_utc.max(),"map_age_reduction_days":float((july_fit.execution_label_end_utc.max()-old_fit.execution_label_end_utc.max())/pd.Timedelta(days=1))}
        write_json(stage/"contract.json",contract); files=[x for x in stage.iterdir() if x.is_file()];manifest={"schema":"july_common30_baseline_map_refresh_v1","status":"SEALED_BASELINE_MAP_REFRESH_COMMON30_LIMITED_NON_PROMOTION","promotion_eligible":False,"inputs":{str((Path(stack)/"manifest.json").resolve()):sha(Path(stack)/"manifest.json"),str((Path(july_root)/"manifest.json").resolve()):sha(Path(july_root)/"manifest.json")},"contract":contract,"outputs_sha256":{x.name:sha(x) for x in files}};write_json(stage/"manifest.json",manifest);(stage/"manifest.sha256").write_text(f"{sha(stage/'manifest.json')}  manifest.json\n");os.replace(stage,output);return output
    except Exception:
        shutil.rmtree(stage,ignore_errors=True);raise


if __name__ == "__main__":
    print(run())
