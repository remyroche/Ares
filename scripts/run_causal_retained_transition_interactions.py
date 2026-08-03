#!/usr/bin/env python3
"""Causal refit and bounded interaction screen for retained transition mechanisms.

This runner is intentionally small: three previously retained logistic
mechanisms, their original targets/features, a fixed three-point bounded
weight grid chosen only on earlier strict-OOF evidence, and global ranking
metrics.  It neither ranks a raw transition probability nor changes entry/
exit actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

try:
    from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES
    from scripts.run_cross_era_regime_transition_classifier_ablation import _model
    from scripts.run_sparse_transition_mechanism_ablation import feature_arms
except ModuleNotFoundError:
    from materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES
    from run_cross_era_regime_transition_classifier_ablation import _model
    from run_sparse_transition_mechanism_ablation import feature_arms


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/pooled_historical_current_transition_panel_20260730_v1/transition_panel.parquet"
GEOMETRY = ROOT / "data_perp/artifacts/forward_exact_transition_geometry_20260730_v1/hourly_geometry.parquet"
FORWARD = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3/forward_predictions.parquet"
MAPPED = ROOT / "data_perp/artifacts/hurdle_cross_side_common_unit_mapping_20260730_v1/mapped_candidates.parquet"
SPARSE = ROOT / "data_perp/artifacts/pooled_historical_current_sparse_transition_mechanism_ablation_20260730_v1/predictions.parquet"
SUPPORT = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3/support_head_oof_ledger.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_retained_transition_interactions_20260730_v2"
WINDOWS = ("may_to_june_forward_control", "later_july_forward")
MECHANISMS = {
    "compression_onset": ("compression_release", "target__adverse_onset_within_3h"),
    "memory_active": ("memory_range_recurrence", "target__active_adverse"),
    "state_active": ("sparse_state_levels", "target__active_adverse"),
}
WEIGHTS = (0.0, 0.25, 0.50)
FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe(x: Any) -> Any:
    if x is pd.NaT or (not isinstance(x, (Mapping, list, tuple)) and pd.isna(x)):
        return None
    if isinstance(x, (Path, pd.Timestamp)): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Mapping): return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_safe(v) for v in x]
    return x


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    order = np.lexsort((frame.candidate_id.astype(str).to_numpy(), -frame[score].to_numpy(float)))
    return frame.iloc[order[:count]].copy()


def tie_metrics(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, Any]:
    values, y = frame[score].to_numpy(float), frame.execution_net_ev_12h.to_numpy(float)
    count = max(1, int(math.ceil(len(frame) * fraction))); cutoff = float(np.sort(values)[-count])
    above, plateau = values > cutoff, values == cutoff; need = count - int(above.sum()); p = y[plateau]
    return {"score_unique_count": int(np.unique(values).size), "cutoff_tie_ambiguous": bool(len(p) > need), "cutoff_plateau_rows": int(len(p)), "tie_expected_net_bps": float((y[above].sum() + need * p.mean()) / count * 1e4)}


def temporal_probability(train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str], target: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit retained logistic recipe and causal fixed-family shrink calibration."""
    local = train.sort_values([f"{target}_available_utc", "cohort_anchor_utc"], kind="stable").reset_index(drop=True)
    y = pd.to_numeric(local[target], errors="raise").astype(int)
    if len(local) < 80 or y.nunique() != 2: raise ValueError(f"insufficient causal support for {target}")
    split = max(40, int(math.floor(0.70 * len(local))))
    early, late = local.iloc[:split], local.iloc[split:]
    y_early, y_late = y.iloc[:split], y.iloc[split:]
    shrink = 1.0
    if len(late) >= 20 and y_early.nunique() == 2 and y_late.nunique() == 2:
        early_model = _model("logistic", columns); early_model.fit(early, y_early)
        raw_late = early_model.predict_proba(late)[:, 1]; prior = float(y_early.mean())
        losses = [brier_score_loss(y_late, np.clip(prior + w * (raw_late - prior), 1e-8, 1 - 1e-8)) for w in WEIGHTS]
        shrink = float(WEIGHTS[int(np.argmin(losses))])
    prior = float(y.mean()); model = _model("logistic", columns); model.fit(local, y)
    raw = model.predict_proba(test)[:, 1]
    return np.clip(prior + shrink * (raw - prior), 1e-8, 1 - 1e-8), {"train_rows": int(len(local)), "positives": int(y.sum()), "prior": prior, "calibration_shrink": shrink, "feature_count": len(columns), "target": target}


def oof_mechanisms(sparse: pd.DataFrame, support: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Join only pre-cutoff strict-OOF mechanism probabilities to OOF economics."""
    prediction = sparse.loc[(sparse.source_family.eq("current_exact_spread_mayjul2026")) & (sparse.evaluation_kind.eq("current_strict_oof_within_source"))].copy()
    pieces = []
    for name, (arm, target) in MECHANISMS.items():
        x = prediction.loc[prediction.arm.eq(arm) & prediction.target_name.eq(target), ["cohort_anchor_utc", "prediction"]].rename(columns={"prediction": name})
        if x.cohort_anchor_utc.duplicated().any(): raise ValueError(f"strict OOF prediction duplicates {name}")
        pieces.append(x)
    result = support.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True)
    result["support_label_available_utc"] = pd.to_datetime(result["support_label_available_utc"], utc=True)
    for item in pieces:
        item["cohort_anchor_utc"] = pd.to_datetime(item["cohort_anchor_utc"], utc=True)
        result = result.merge(item, left_on="__ts__", right_on="cohort_anchor_utc", how="left", validate="many_to_one").drop(columns="cohort_anchor_utc")
    return result.loc[result.support_label_available_utc.lt(cutoff)].copy()


def select_weights(oof: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[dict[str, float], pd.DataFrame]:
    """Choose bounded weights solely from earlier strict-OOF net economics."""
    usable = oof.dropna(subset=["side_causal_oof_ev_gross_cost_hurdle_ev", "p_gross_exceeds_cost", *MECHANISMS]).copy()
    usable = usable.loc[np.isfinite(usable.execution_net_ev_12h) & np.isfinite(usable.side_causal_oof_ev_gross_cost_hurdle_ev)].copy()
    if len(usable) < 500: raise ValueError("insufficient joined pre-cutoff strict OOF support for interaction selection")
    base = usable.side_causal_oof_ev_gross_cost_hurdle_ev.to_numpy(float); scale = float(np.std(base))
    if scale <= 0: raise ValueError("OOF hurdle score has zero scale")
    risk = usable[list(MECHANISMS)].mean(axis=1).to_numpy(float); conv = usable.p_gross_exceeds_cost.to_numpy(float)
    recipes = {"uncertainty_penalty": lambda w: base - w * scale * risk, "hurdle_conversion_interaction": lambda w: base + w * scale * (conv - conv.mean()) * (1.0 - risk)}
    audit = [] ; selected = {}
    for name, fn in recipes.items():
        best_weight, best_value = None, -np.inf
        for w in WEIGHTS:
            work = usable.assign(score=fn(w)); value = float(stable_top(work, "score", 0.10).execution_net_ev_12h.mean())
            audit.append({"cutoff_utc": cutoff, "recipe": name, "weight": w, "selection_rows": len(work), "oof_top10_net_bps": value * 1e4})
            if value > best_value: best_weight, best_value = float(w), value
        selected[name] = float(best_weight)
    return selected, pd.DataFrame(audit)


def scores(frame: pd.DataFrame, *, scale: float, weights: Mapping[str, float], priors: Mapping[str, float]) -> pd.DataFrame:
    result = frame.copy(); risk = result[list(MECHANISMS)].mean(axis=1).to_numpy(float)
    base = result.mapped_score.to_numpy(float); conv = result.p_gross_exceeds_cost.to_numpy(float)
    result["control_mapped_hurdle"] = base
    result["uncertainty_penalty"] = base - weights["uncertainty_penalty"] * scale * risk
    result["hurdle_conversion_interaction"] = base + weights["hurdle_conversion_interaction"] * scale * (conv - priors["conversion"]) * (1.0 - risk)
    # Individual retained mechanisms remain interpretable diagnostics, not a
    # probability ranker.  Their fixed penalty magnitude is selected above.
    for name in MECHANISMS:
        result[f"penalty_{name}"] = base - weights["uncertainty_penalty"] * scale * result[name].to_numpy(float)
    return result


def evaluate(frame: pd.DataFrame, window: str, score_columns: Sequence[str]) -> pd.DataFrame:
    rows=[]; week = frame.__ts__.dt.normalize() - pd.to_timedelta(frame.__ts__.dt.dayofweek, unit="D")
    for scope, local in (("all", frame), ("latest_week", frame.loc[week.eq(week.max())])):
        for score in score_columns:
            for fraction in FRACTIONS:
                chosen=stable_top(local,score,fraction); assets=chosen.__symbol__.value_counts(normalize=True)
                rows.append({"window":window,"scope":scope,"score_arm":score,"fraction":fraction,"candidate_rows":len(local),"selected_rows":len(chosen),"mean_net_bps":float(chosen.execution_net_ev_12h.mean()*1e4),"positive_net_rate":float(chosen.execution_net_ev_12h.gt(0).mean()),"long_share":float(chosen.side_name.eq('long').mean()),"asset_count":int(chosen.__symbol__.nunique()),"asset_top_share":float(assets.iloc[0]),"asset_hhi":float((assets**2).sum()),**tie_metrics(local,score,fraction)})
    return pd.DataFrame(rows)


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"immutable output exists: {output_dir}")
    panel, geom, forward, mapped, sparse, support = (pd.read_parquet(p) for p in (PANEL, GEOMETRY, FORWARD, MAPPED, SPARSE, SUPPORT))
    features=feature_arms(CANONICAL_FEATURES); panel=panel.loc[(panel.source_family.eq("current_exact_spread_mayjul2026")) & panel.mapping_provenance_role.eq("strict_oof") & panel.context_available.astype(bool)].copy()
    for c in ["cohort_anchor_utc", *[f"{target}_available_utc" for _, target in MECHANISMS.values()]]: panel[c]=pd.to_datetime(panel[c],utc=True)
    geom["signal_context_utc"]=pd.to_datetime(geom.signal_context_utc,utc=True); forward["__ts__"]=pd.to_datetime(forward.__ts__,utc=True)
    mapped=mapped.loc[mapped.map_arm.eq("pooled_plus_side_residual_shrink_4000"), ["candidate_id","mapped_score"]].copy()
    all_metrics=[]; all_scores=[]; all_fit=[]; all_select=[]
    for window in WINDOWS:
        target=forward.loc[forward.window.eq(window)].copy(); cutoff=target.__ts__.min()
        target=target.merge(geom.drop(columns="common_transition_context_available"),left_on="__ts__",right_on="signal_context_utc",how="left",validate="many_to_one").merge(mapped,on="candidate_id",how="left",validate="one_to_one")
        if target[list(CANONICAL_FEATURES)].notna().any(axis=1).mean()!=1.0 or target.mapped_score.isna().any(): raise ValueError(f"incomplete geometry/map in {window}")
        for name,(arm,label) in MECHANISMS.items():
            train=panel.loc[panel[f"{label}_available_utc"].lt(cutoff) & panel[label].notna()].copy()
            pred,audit=temporal_probability(train,target,features[arm],label); target[name]=pred; audit.update({"window":window,"mechanism":name,"cutoff_utc":cutoff}); all_fit.append(audit)
        selected, selection=select_weights(oof_mechanisms(sparse,support,cutoff),cutoff); selection["window"]=window; all_select.append(selection)
        scale=float(np.std(target.mapped_score.to_numpy(float))); priors={"conversion":float(target.p_gross_exceeds_cost.mean())}; target=scores(target,scale=scale,weights=selected,priors=priors)
        cols=["control_mapped_hurdle","uncertainty_penalty","hurdle_conversion_interaction",*[f"penalty_{x}" for x in MECHANISMS]]
        target["window"]=window; all_scores.append(target); all_metrics.append(evaluate(target,window,cols))
    score_frame=pd.concat(all_scores,ignore_index=True); metrics=pd.concat(all_metrics,ignore_index=True); fit=pd.DataFrame(all_fit); select=pd.concat(all_select,ignore_index=True)
    controls=metrics.loc[(metrics.score_arm.eq("control_mapped_hurdle")) & metrics.fraction.eq(.10)]; gate=bool((controls.mean_net_bps.gt(0)).all() and controls.loc[controls.scope.eq("latest_week"),"mean_net_bps"].gt(0).all() and controls.long_share.between(.05,.95).all())
    stage=Path(tempfile.mkdtemp(dir=output_dir.parent,prefix=f".{output_dir.name}."))
    try:
        for n,x in (("scores.parquet",score_frame),("metrics.csv",metrics),("causal_refit_audit.csv",fit),("weight_selection_oof.csv",select)):
            (x.to_parquet(stage/n,index=False,compression="zstd") if n.endswith("parquet") else x.to_csv(stage/n,index=False))
        manifest={"schema":"causal_retained_transition_interactions_v1","status":"CAUSAL_REFIT_BOUNDED_INTERACTION_SCREEN_COMPLETE","promotion_eligible":False,"policy_portfolio_ran":False,"policy_gate_passed":gate,"mechanisms":MECHANISMS,"weights":list(WEIGHTS),"contracts":{"refit":"per-window causal refit; every mechanism label resolves strictly before that window cutoff; original retained logistic feature/target recipe; temporal fixed-family Brier shrink calibration","weight_selection":"only earlier strict-OOF hurdle/support rows; 0/.25/.50 bounded grid; no HPO","ranking":"one pooled global score across timestamps/sides after causal common-unit hurdle mapping; no raw transition-probability ranking, side quota or action-layer input","economics":"exact frozen 12h net outcome and causal mapped hurdle score"},"source_hashes":{str(p):sha256(p) for p in (PANEL,GEOMETRY,FORWARD,MAPPED,SPARSE,SUPPORT)},"outputs_sha256":{n:sha256(stage/n) for n in ("scores.parquet","metrics.csv","causal_refit_audit.csv","weight_selection_oof.csv")},"runner":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())}}
        write_json(stage/"manifest.json",manifest); (stage/"manifest.sha256").write_text(sha256(stage/"manifest.json")+"\n"); os.replace(stage,output_dir)
    except Exception:
        import shutil; shutil.rmtree(stage,ignore_errors=True); raise
    return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args(); print(json.dumps(_safe(run(a.output_dir)),sort_keys=True))
