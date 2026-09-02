#!/usr/bin/env python3
"""Phase E/F conversion objectives on frozen strict-OOF O250/H6 probabilities.

The script changes only the conditional conversion head trained on true
short O250/H6 rows.  It compares the existing five-state normalized-regret
multiclass C3 with cumulative and continuation-ratio ordinal formulations,
median/upper quantile normalized-regret regressors, their small conservative
composite, and a Huber normalized-regret control.  The opportunity score is
the existing strict-OOF C59 O45 score in every arm; K0 stays analytic and the
admission floor remains +75 bps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402
import run_strict_r3_short_p0_oc_k0_phase_a_timing as phase_a  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_ef_conversion_v1"
STATIC_C59 = phase_a.STATIC_C59
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_ef_conversion_202408_202607_20260822_v1"
SEED = 1729
MIN_C_ROWS = r3.MIN_C_ROWS
MIN_OOF_ROWS = phase_a.MIN_OOF_ROWS
MIN_OOF_MONTHS = phase_a.MIN_OOF_MONTHS
ADMISSION_BPS = phase_a.ADMISSION_BPS
POLICY_CLIP_BPS = phase_a.POLICY_CLIP_BPS


@dataclass(frozen=True)
class CSpec:
    name: str
    kind: str  # multiclass | cumulative | continuation | quantile | huber
    alpha: float | None = None
    penalty: float | None = None
    description: str = ""


SPECS = (
    CSpec("E0_C3_multiclass_control", "multiclass", description="current 5-state normalized-regret C3"),
    CSpec("E1_cumulative_ordinal", "cumulative", description="P(C>k), k=0..3, monotonic projection"),
    CSpec("E2_continuation_ratio", "continuation", description="sequential P(C>=k | C>=k-1)"),
    CSpec("F1_q50_normalized_regret", "quantile", .50, description="-q50 normalized regret"),
    CSpec("F2_q75_normalized_regret", "quantile", .75, description="-q75 normalized regret"),
    CSpec("F3_q25_q50_q75_l025", "quantile", .50, .25, description="-q50 - .25*(q75-q50)"),
    CSpec("F3_q25_q50_q75_l050", "quantile", .50, .50, description="-q50 - .50*(q75-q50)"),
    CSpec("F3_q25_q50_q75_l100", "quantile", .50, 1.00, description="-q50 - 1.00*(q75-q50)"),
    CSpec("F4_huber_normalized_regret", "huber", description="negative Huber normalized regret"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r1._valid_label(frame)


def _event(frame: pd.DataFrame) -> np.ndarray:
    return r1._event(frame, r3.SPEC)


def _grade(frame: pd.DataFrame) -> np.ndarray:
    return r3._target(frame, next(item for item in r3.TARGETS if item.name == "C3_normalized_regret"))


def _regret(frame: pd.DataFrame) -> np.ndarray:
    return r3._norm_regret(frame)


def _load() -> tuple[pd.DataFrame, tuple[str, ...], dict[str, str]]:
    frame, _o, _m4, hashes = r3._load_frame()
    fields = c59._c59()
    if any(field not in frame for field in fields):
        raise AssertionError("C59 feature missing from target-free short frame")
    static = pd.read_parquet(STATIC_C59, columns=["candidate_id", "__decision_ts__", "opportunity_raw_score", "conversion_score", "held_month"])
    static["__decision_ts__"] = r1._utc(static["__decision_ts__"])
    if static.candidate_id.duplicated().any():
        raise AssertionError("frozen C59 outer identities are non-unique")
    static = static.rename(columns={"opportunity_raw_score": "frozen_o_raw", "conversion_score": "frozen_c59_score", "held_month": "static_held_month"})
    frame = frame.merge(static, on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    return frame, tuple(fields), {**hashes, "static_c59": _sha256(STATIC_C59)}


def _common(seed: int) -> dict[str, Any]:
    return dict(n_estimators=180, learning_rate=.035, max_depth=3, num_leaves=15, min_child_samples=40, subsample=.85, subsample_freq=1, colsample_bytree=.85, reg_lambda=4.0, reg_alpha=.10, random_state=seed, n_jobs=-1, verbosity=-1)


def _fit(spec: CSpec, x: pd.DataFrame, grade: np.ndarray, regret: np.ndarray, seed: int) -> object:
    params = _common(seed)
    if spec.kind == "multiclass":
        model = LGBMClassifier(objective="multiclass", num_class=5, class_weight="balanced", **params)
        model.fit(x, grade)
        return model
    if spec.kind in ("cumulative", "continuation"):
        models: list[LGBMClassifier] = []
        for level in range(4):
            if spec.kind == "cumulative":
                mask = np.ones(len(grade), dtype=bool)
                target = (grade > level).astype(np.int8)
            else:
                mask = grade >= level
                target = (grade[mask] >= level + 1).astype(np.int8)
            if mask.sum() < 100 or np.unique(target).size < 2:
                raise ValueError("insufficient ordinal threshold support")
            model = LGBMClassifier(objective="binary", class_weight="balanced", **_common(seed + level))
            model.fit(x.loc[mask], target)
            models.append(model)
        return tuple(models)
    if spec.kind == "quantile":
        alpha = float(spec.alpha)
        if spec.penalty is None:
            model = LGBMRegressor(objective="quantile", alpha=alpha, **params)
            model.fit(x, regret)
            return model
        models = []
        for index, alpha in enumerate((.25, .50, .75)):
            model = LGBMRegressor(objective="quantile", alpha=alpha, **_common(seed + index))
            model.fit(x, regret)
            models.append(model)
        return tuple(models)
    if spec.kind == "huber":
        model = LGBMRegressor(objective="huber", alpha=.90, **params)
        model.fit(x, regret)
        return model
    raise ValueError(spec.kind)


def _predict(model: object, spec: CSpec, x: pd.DataFrame) -> np.ndarray:
    if spec.kind == "multiclass":
        p = model.predict_proba(x)  # type: ignore[union-attr]
        return np.asarray(p @ np.arange(5, dtype=float), dtype=float)
    if spec.kind == "cumulative":
        p = np.column_stack([part.predict_proba(x)[:, 1] for part in model])  # type: ignore[union-attr]
        # P(C>0) >= ... >= P(C>3).
        return np.minimum.accumulate(np.clip(p, 0.0, 1.0), axis=1).sum(axis=1)
    if spec.kind == "continuation":
        conditionals = np.column_stack([part.predict_proba(x)[:, 1] for part in model])  # type: ignore[union-attr]
        cumulative = np.cumprod(np.clip(conditionals, 0.0, 1.0), axis=1)
        return cumulative.sum(axis=1)
    if spec.kind == "quantile":
        if spec.penalty is None:
            return -np.asarray(model.predict(x), dtype=float)  # type: ignore[union-attr]
        q25, q50, q75 = [np.asarray(part.predict(x), dtype=float) for part in model]  # type: ignore[union-attr]
        _ = q25  # fitted/reportable calibration diagnostic; the stated score uses q50/q75.
        return -q50 - float(spec.penalty) * np.maximum(q75 - q50, 0.0)
    if spec.kind == "huber":
        return -np.asarray(model.predict(x), dtype=float)  # type: ignore[union-attr]
    raise ValueError(spec.kind)


def _inner_c(train: pd.DataFrame, fields: tuple[str, ...], spec: CSpec, held_month: str, seed: int) -> pd.DataFrame:
    local = train.loc[_valid(train) & train.frozen_o_raw.notna()].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    edges = np.linspace(0, len(local), r1.INNER_SPLITS + 2, dtype=int)
    parts: list[pd.DataFrame] = []
    for fold in range(r1.INNER_SPLITS):
        valid = local.iloc[int(edges[fold + 1]):int(edges[fold + 2])].copy()
        if valid.empty:
            continue
        cutoff = valid.__decision_ts__.min()
        fit = local.loc[local.__label_available_at__.lt(cutoff)].copy()
        c_fit = fit.loc[_event(fit).astype(bool)].copy()
        if len(c_fit) < MIN_C_ROWS:
            continue
        x_fit, med = r1._matrix(c_fit, fields); x_valid, _ = r1._matrix(valid, fields, med)
        model = _fit(spec, x_fit, _grade(c_fit), _regret(c_fit), seed + fold * 100)
        part = valid.loc[:, ["candidate_id", "frozen_o_raw", "policy_net_bps", "__decision_ts__"]].copy()
        part["conversion_oof_raw"] = _predict(model, spec, x_valid).astype(np.float32)
        part["opp_oof_raw"] = part.pop("frozen_o_raw").to_numpy(np.float32)
        part["event_target"] = _event(valid).astype(np.int8)
        part["held_month"] = held_month
        parts.append(part)
    if not parts:
        raise ValueError("no purged inner C OOF support")
    out = pd.concat(parts, ignore_index=True)
    if len(out) < MIN_OOF_ROWS or out.__decision_ts__.dt.strftime("%Y-%m").nunique() < MIN_OOF_MONTHS:
        raise ValueError("insufficient combined O/C OOF support")
    return out


def _outer_month(frame: pd.DataFrame, fields: tuple[str, ...], spec: CSpec, month: pd.Timestamp, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    end = month + pd.offsets.MonthBegin(1)
    held = frame.loc[frame.__decision_ts__.ge(month) & frame.__decision_ts__.lt(end)].copy()
    train = frame.loc[frame.__decision_ts__.lt(month) & frame.__label_available_at__.lt(month) & _valid(frame)].copy()
    if held.empty or held.frozen_o_raw.isna().any():
        raise ValueError("frozen O45 OOF score unavailable for held month")
    inner = _inner_c(train, fields, spec, month.strftime("%Y-%m"), seed)
    bundle = phase_a._fit_k0(inner)
    c_train = train.loc[_event(train).astype(bool)].copy()
    if len(c_train) < MIN_C_ROWS:
        raise ValueError("insufficient outer C support")
    x_train, med = r1._matrix(c_train, fields); x_held, _ = r1._matrix(held, fields, med)
    model = _fit(spec, x_train, _grade(c_train), _regret(c_train), seed + 10_000)
    c_score = _predict(model, spec, x_held)
    out = held.loc[:, [*r1.IDENTITY, "__label_available_at__", "policy_net_bps", "policy_regret_bps", "policy_gross_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid", r3.SPEC.label_field]].copy().reset_index(drop=True)
    out["opportunity_raw_score"] = held.frozen_o_raw.to_numpy(np.float32)
    out = pd.concat((out, phase_a._apply_k0(bundle, held.frozen_o_raw.to_numpy(float), c_score)), axis=1)
    out["held_month"] = month.strftime("%Y-%m"); out["arm"] = spec.name
    return out, {"arm": spec.name, "held_month": month.strftime("%Y-%m"), "status": "complete", "outer_train_rows": len(train), "outer_c_rows": len(c_train), "inner_oof_rows": bundle.oof_rows, "inner_oof_months": bundle.oof_months, "k0_mu0_bps": bundle.mu0}


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 5 or np.unique(left).size < 2 or np.unique(right).size < 2: return float("nan")
    return float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def _c_metrics(part: pd.DataFrame) -> dict[str, Any]:
    valid = part.loc[_valid(part)].copy()
    event = _event(valid).astype(bool); c = valid.conversion_score.to_numpy(float)
    target_grade = _grade(valid)
    net = r1._finite(valid.policy_net_bps).to_numpy(float)
    if event.sum():
        grade = target_grade[event]; score = c[event]; outcome = net[event]
        predicted_grade = np.clip(np.rint(score), 0, 4).astype(int)
        kappa = float(cohen_kappa_score(grade, predicted_grade, weights="quadratic")) if len(np.unique(grade)) > 1 else float("nan")
        mae = float(mean_absolute_error(grade, np.clip(score, 0, 4)))
        rank_ic = _spearman(score, -_regret(valid)[event])
        net_ic = _spearman(score, outcome)
        bins = pd.qcut(pd.Series(score).rank(method="first"), q=min(5, len(score)), labels=False, duplicates="drop")
        means = pd.DataFrame({"bin": bins, "net": outcome}).groupby("bin", observed=False).net.mean().to_numpy(float)
        monotonic = float(np.mean(np.diff(means) >= 0.0)) if len(means) > 1 else float("nan")
    else:
        kappa = mae = rank_ic = net_ic = monotonic = float("nan")
    admitted = part.loc[pd.to_numeric(part.K0_expected_policy_net_bps, errors="coerce").ge(ADMISSION_BPS)]
    known = admitted.loc[_valid(admitted)]; values = r1._finite(known.policy_net_bps).to_numpy(float)
    return {"arm": str(part.arm.iloc[0]), "held_month": str(part.held_month.iloc[0]), "valid_rows": len(valid), "conditional_rows": int(event.sum()), "c_rank_ic": rank_ic, "c_net_spearman": net_ic, "quadratic_kappa": kappa, "ordinal_mae": mae, "mu1_monotonic_step_fraction": monotonic, "admitted": len(admitted), "known_admitted": len(known), "net_bps_per_trade": float(values.mean()) if len(values) else float("nan"), "total_net_bps": float(values.sum()) if len(values) else 0.0, "cvar10_bps": r1._cvar(values), "p_net_lt_neg200": float((values < -200).mean()) if len(values) else float("nan"), "positive_fraction": float((values > 0).mean()) if len(values) else float("nan")}


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); weights = np.asarray(weights, dtype=float); keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep])) if keep.any() else float("nan")


def _era(monthly: pd.DataFrame) -> pd.DataFrame:
    result=[]
    for (arm, era), g in monthly.assign(era=monthly.held_month.str[:4]).groupby(["arm", "era"], sort=True):
        w=np.maximum(g.known_admitted.to_numpy(float),1.0); row={"arm":arm,"era":era,"months":len(g),"admitted":int(g.admitted.sum()),"known_admitted":int(g.known_admitted.sum()),"total_net_bps":float(g.total_net_bps.sum()),"positive_months":int((g.net_bps_per_trade>0).sum()),"worst_month_net_bps":float(g.net_bps_per_trade.min())}
        for name in ("c_rank_ic","c_net_spearman","quadratic_kappa","ordinal_mae","mu1_monotonic_step_fraction","net_bps_per_trade","cvar10_bps","p_net_lt_neg200","positive_fraction"):
            row[name]=_weighted(g[name].to_numpy(float),w)
        result.append(row)
    return pd.DataFrame(result)


def _summary(monthly: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for arm,g in era.loc[era.era.isin(("2025","2026"))].groupby("arm",sort=True):
        by=g.set_index("era"); w=np.maximum(g.known_admitted.to_numpy(float),1.0); m=monthly.loc[monthly.arm.eq(arm)&monthly.held_month.str[:4].isin(("2025","2026"))]
        rows.append({"arm":arm,"net_2025":float(by.loc["2025","net_bps_per_trade"]) if "2025" in by.index else float("nan"),"net_2026":float(by.loc["2026","net_bps_per_trade"]) if "2026" in by.index else float("nan"),"mean_net_bps":_weighted(g.net_bps_per_trade.to_numpy(float),w),"total_net_bps":float(g.total_net_bps.sum()),"worst_era_net_bps":float(g.net_bps_per_trade.min()),"worst_month_net_bps":float(m.net_bps_per_trade.min()),"c_rank_ic":_weighted(g.c_rank_ic.to_numpy(float),w),"c_net_spearman":_weighted(g.c_net_spearman.to_numpy(float),w),"mu1_monotonic":_weighted(g.mu1_monotonic_step_fraction.to_numpy(float),w),"cvar10_bps":_weighted(g.cvar10_bps.to_numpy(float),w),"admitted":int(g.admitted.sum())})
    out=pd.DataFrame(rows); out["advances_phase_ef"]=out.net_2025.ge(90)&out.net_2026.ge(90)&out.c_rank_ic.gt(0)
    return out.sort_values(["mean_net_bps","worst_month_net_bps","c_rank_ic"],ascending=False,kind="stable").reset_index(drop=True)


def _table(frame: pd.DataFrame)->str:
    try:return frame.to_markdown(index=False)
    except ImportError:
        c=list(map(str,frame.columns)); return "\n".join(["| "+" | ".join(c)+" |","| "+" | ".join("---" for _ in c)+" |",*("| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))])


def run(out:Path,specs:tuple[CSpec,...],months:pd.DatetimeIndex)->Path:
    if out.exists():raise FileExistsError(out)
    frame,fields,hashes=_load(); preds=[]; audits=[]; metrics=[]
    for si,spec in enumerate(specs):
        for mi,month in enumerate(months):
            try:
                p,a=_outer_month(frame,fields,spec,month,SEED+si*20000+mi*101);preds.append(p);audits.append(a);metrics.append(_c_metrics(p))
            except ValueError as e:audits.append({"arm":spec.name,"held_month":month.strftime("%Y-%m"),"status":"skipped","reason":str(e)})
    if not preds:raise RuntimeError("no Phase E/F predictions")
    monthly=pd.DataFrame(metrics);era=_era(monthly);summary=_summary(monthly,era);out.mkdir(parents=True)
    pd.concat(preds,ignore_index=True).to_parquet(out/"phase_ef_outer_oof_predictions.parquet",index=False,compression="zstd");pd.DataFrame(audits).to_parquet(out/"phase_ef_fold_audit.parquet",index=False,compression="zstd");monthly.to_parquet(out/"phase_ef_monthly_metrics.parquet",index=False,compression="zstd");era.to_parquet(out/"phase_ef_era_metrics.parquet",index=False,compression="zstd");summary.to_parquet(out/"phase_ef_summary.parquet",index=False,compression="zstd")
    manifest={"schema":SCHEMA,"status":"complete","side":"short","scope":"Phase E/F only; no canonical/live change","period":{"candidate_start":"2024-05","output_supported_start":"2024-10","end_exclusive":"2026-08"},"specs":[s.__dict__ for s in specs],"opportunity":"frozen O250/H6 strict-OOF C59 O45 scores","conversion_features":list(fields),"admission":{"K0_expected_policy_net_bps_gte":ADMISSION_BPS},"causality":{"C_fit":"true O-positive labels resolved before inner/outer score cutoffs","O":"existing strict OOF C59 ledger","candidates":"target-free held candidates all scored","invalidity":"invalid paths excluded only from fitting/outcome metrics"},"sources":hashes}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");(out/"SHORT_P0_OC_K0_PHASE_EF_CONVERSION_REPORT.md").write_text("\n".join(["# Short P0 → frozen O → conditional C → K0: Phase E/F","","Research-only, strict prequential 2024/2025/2026 scorecard.","","## Era metrics","",_table(era),"","## Sequential selection","",_table(summary),"","```json",json.dumps(manifest,indent=2),"```",""]))
    return out


def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,default=OUT);p.add_argument("--months",nargs="*",default=None);p.add_argument("--spec",action="append",default=[]);a=p.parse_args()
    months=pd.DatetimeIndex([pd.Timestamp(x+"-01T00:00:00Z") for x in a.months]) if a.months else pd.date_range("2024-05-01T00:00:00Z","2026-08-01T00:00:00Z",freq="MS",inclusive="left")
    lookup={s.name:s for s in SPECS};specs=tuple(lookup[x] for x in a.spec) if a.spec else SPECS
    print(run(a.out,specs,months))


if __name__=="__main__":main()
