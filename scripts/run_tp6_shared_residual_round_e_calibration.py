#!/usr/bin/env python3
"""Round E: causal common-bps calibration of the D shared residual stacks.

This is deliberately a calibration-only replay.  It never refits D0/C2 or D3,
does not create a local or regime expert, and does not apply an admission or
policy gate.  For each held-out outer era it fits the additive calibrator on
*earlier held-out forecasts* whose H12 outcomes have already resolved, then
ranks the whole current era globally in the resulting common bps units.

E0: global additive correction.
E1: global + strongly shrunk side correction.
E2: E1 + strongly shrunk side x soft-regime additive correction.
E3: strongly shrunk global/side/side x soft-regime affine correction.  The
    regime term remains an expectation over the contemporaneous soft simplex,
    never a routed regime model.

The initial outer era intentionally uses identity mapping: no earlier OOF
forecast exists from which a calibration relation can honestly be learnt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.shared_regime_calibration import (
    CausalCalibrationError,
    fit_shared_bps_calibration,
    predict_shared_bps_calibration,
)

D_ROOT = ROOT / "data_perp/artifacts/tp6_shared_residual_d0_d4_20260809_v1"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/tp6_shared_residual_round_e_calibration_20260809_v1"
ERAS = ("2023-09_10", "2023-11_12", "2024-01_02", "2024-05_06", "2024-07_08", "2024-09_10", "2024-11")
SOURCE_ARMS = ("D0_c2_control", "D3_active_failure_probability")
MODES = (
    "C0_global",
    "C1_side",
    "C2_side_soft_regime",
    "C3_hierarchical_affine_soft_regime",
)
SOFT = ("regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition")
TOPS = (.01, .05, .10)
# Availability is not the raw path horizon: predictions are made at signal
# close, entry is +1h, and the H12 outcome becomes known at +13h.
LABEL_AVAILABILITY_DELAY = pd.Timedelta(hours=13)
# Deliberately conservative: side and especially soft-state corrections are
# small departures from the globally comparable bps score.
SIDE_SHRINK = 3_000.0
REGIME_SHRINK = 6_000.0
REGIME_CAP = .25
MIN_ROWS = 500


def _rank_metrics(frame: pd.DataFrame, score: np.ndarray, common: dict[str, Any], *, scope: str, period: str) -> list[dict[str, Any]]:
    x = frame.copy(); x["calibrated_bps"] = score
    rows: list[dict[str, Any]] = []
    for view, q in (("global", x), ("long", x[x.side_name.eq("long")]), ("short", x[x.side_name.eq("short")])):
        if q.empty:
            continue
        ic = spearmanr(q.calibrated_bps, q.net_bps).statistic if len(q) > 1 else np.nan
        for top in TOPS:
            take = q.sort_values(["calibrated_bps", "candidate_id"], ascending=[False, True], kind="stable").head(max(1, int(np.ceil(len(q) * top))))
            rows.append({**common, "scope": scope, "period": period, "view": view, "top_fraction": top,
                         "eligible_rows": len(q), "selected_rows": len(take),
                         "net_bps": float(take.net_bps.mean()), "gross_bps": float(take.gross_bps.mean()),
                         "score_net_spearman": float(ic), "selected_long_fraction": float(take.side_name.eq("long").mean()),
                         "positive_net_fraction": float(take.net_bps.gt(0).mean())})
    return rows


def _ensure_label_available_ts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "label_available_ts" in out:
        available = pd.to_datetime(out["label_available_ts"], utc=True, errors="raise")
    elif "__label_available_at__" in out:
        available = pd.to_datetime(out["__label_available_at__"], utc=True, errors="raise")
    else:
        available = pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY
    if available.isna().any() or (available < pd.to_datetime(out["__ts__"], utc=True) + LABEL_AVAILABILITY_DELAY).any():
        raise ValueError("H12 label availability must be signal-close +13h or later")
    out["label_available_ts"] = available
    return out


def _load() -> pd.DataFrame:
    pieces = []
    for path in sorted((D_ROOT.with_name(D_ROOT.name + "_stage") / "predictions").glob("*.parquet")):
        # Each checkpoint is one arm.  Avoid deserialising the three rejected
        # D arms merely to filter them afterwards; that needlessly multiplies
        # memory and defeats the deliberately bounded Round-E replay.
        if not any(path.stem.endswith("_" + arm) for arm in SOURCE_ARMS):
            continue
        x = pd.read_parquet(path)
        pieces.append(x)
    if not pieces:
        raise FileNotFoundError("Round-D prediction checkpoints are missing")
    x = pd.concat(pieces, ignore_index=True)
    if x.duplicated(["candidate_id", "arm"]).any():
        raise ValueError("duplicate Round-D out-of-era predictions")
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    # Soft state is decision-time data from the sealed common ledger.  It is
    # joined only by candidate identity; no later label fields are imported.
    state = pd.read_parquet(LEDGER, columns=["candidate_id", *SOFT, "label_available_ts"])
    if state.candidate_id.duplicated().any():
        raise ValueError("nonunique sealed soft-state candidate identity")
    x = x.merge(state, on="candidate_id", how="left", validate="many_to_one", suffixes=("", "_ledger"))
    if "label_available_ts_ledger" in x:
        if "label_available_ts" in x:
            supplied = pd.to_datetime(x["label_available_ts"], utc=True, errors="raise")
            ledger = pd.to_datetime(x["label_available_ts_ledger"], utc=True, errors="raise")
            if not supplied.eq(ledger).all():
                raise ValueError("prediction and sealed-ledger label availability disagree")
            x = x.drop(columns="label_available_ts_ledger")
        else:
            x = x.rename(columns={"label_available_ts_ledger": "label_available_ts"})
    x = _ensure_label_available_ts(x)
    x["outcome_resolved_at"] = x["label_available_ts"]
    if x.loc[:, SOFT].isna().any().any() or not np.allclose(x.loc[:, SOFT].sum(axis=1), 1., atol=1e-6):
        raise ValueError("missing or invalid sealed soft-regime state")
    if not np.allclose(x.gross_bps.to_numpy(float) - x.net_bps.to_numpy(float), 100., atol=.02):
        raise ValueError("fixed 100-bps contract failed")
    if set(x.era.unique()) != set(ERAS):
        raise ValueError(f"unexpected outer-era population: {sorted(x.era.unique())}")
    return x.sort_values(["__ts__", "candidate_id", "arm"], kind="stable").reset_index(drop=True)


def _calibrate(prior: pd.DataFrame, test: pd.DataFrame, mode: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    cutoff = test.__ts__.min()
    resolved = prior[prior.outcome_resolved_at.lt(cutoff)].copy()
    if len(resolved) and not pd.to_datetime(resolved["label_available_ts"], utc=True).max() < cutoff:
        raise RuntimeError("calibration prior includes a label not available before its fit cutoff")
    if len(resolved) < MIN_ROWS:
        detail = pd.DataFrame({"raw_common_bps": test.score_bps.to_numpy(float), "calibrated_common_bps": test.score_bps.to_numpy(float),
                               "calibration_global_correction_bps": 0., "calibration_side_correction_bps": 0.,
                               "calibration_soft_regime_correction_bps": 0., "calibration_mode": mode,
                               "calibration_fit_before_utc": cutoff, "calibration_max_resolution_utc": pd.NaT}, index=test.index)
        return test.score_bps.to_numpy(float), detail, {"status": "identity_insufficient_prior_oof", "prior_rows": int(len(resolved)), "fit_before_utc": str(cutoff)}
    try:
        cal = fit_shared_bps_calibration(
            resolved, resolved.score_bps.to_numpy(float), resolved.net_bps.to_numpy(float),
            fit_before_utc=cutoff, mode=mode, soft_regime_columns=SOFT,
            min_global_rows=MIN_ROWS, side_shrink_rows=SIDE_SHRINK,
            regime_shrink_rows=REGIME_SHRINK, regime_weight_cap=REGIME_CAP,
        )
        detail = predict_shared_bps_calibration(cal, test, test.score_bps.to_numpy(float), return_details=True)
        return detail.calibrated_common_bps.to_numpy(float), detail, {
            "status": "prior_resolved_hierarchical_calibration", "prior_rows": int(len(resolved)),
            "fit_before_utc": str(cutoff), "max_resolution_utc": str(cal.max_resolution_utc),
            "global_correction_bps": cal.global_correction_bps, "side_support": cal.side_support,
            "regime_effective_support": cal.regime_effective_support,
        }
    except CausalCalibrationError as exc:
        raise RuntimeError(f"calibration contract failure for {mode} at {cutoff}: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only-source-arm", choices=SOURCE_ARMS, help="bounded one-stack replay cell")
    ap.add_argument("--only-era", choices=ERAS, help="bounded one-held-out-era replay cell")
    args = ap.parse_args()
    if bool(args.only_source_arm) != bool(args.only_era):
        ap.error("--only-source-arm and --only-era must be supplied together")
    if args.out.exists():
        raise FileExistsError(args.out)
    x = _load(); metrics: list[dict[str, Any]] = []; pred: list[pd.DataFrame] = []; audits: list[dict[str, Any]] = []
    source_arms = (args.only_source_arm,) if args.only_source_arm else SOURCE_ARMS
    eras = (args.only_era,) if args.only_era else ERAS
    for arm in source_arms:
        source = x[x.arm.eq(arm)].copy()
        for era in eras:
            i = ERAS.index(era)
            test = source[source.era.eq(era)].copy()
            prior = source[source.era.isin(ERAS[:i])].copy()
            if test.empty:
                raise ValueError(f"missing {arm}/{era}")
            for mode in MODES:
                score, detail, audit = _calibrate(prior, test, mode)
                score_by_row = pd.Series(score, index=test.index, dtype=float)
                common = {"source_arm": arm, "calibration_mode": mode, "test_era": era, "prior_outer_eras": list(ERAS[:i]), "prior_rows": int(len(prior))}
                metrics += _rank_metrics(test, score, common, scope="outer_era", period=era)
                for month, q in test.assign(_month=test.__ts__.dt.strftime("%Y-%m")).groupby("_month", sort=True):
                    metrics += _rank_metrics(q, score_by_row.loc[q.index].to_numpy(float), common, scope="month", period=month)
                p = test[["candidate_id", "__ts__", "side_name", "era", "net_bps", "gross_bps", "score_bps", *SOFT]].copy()
                p["source_arm"] = arm; p["calibration_mode"] = mode; p["calibrated_bps"] = score
                for col in detail.columns:
                    if col not in p:
                        p[col] = detail[col].to_numpy()
                pred.append(p); audits.append({**common, **audit})
            print(f"calibrated {arm} {era}", flush=True)
    mm = pd.DataFrame(metrics); pp = pd.concat(pred, ignore_index=True); aa = pd.DataFrame(audits)
    summary = (mm[(mm.scope.eq("outer_era")) & (mm.view.eq("global")) & (mm.top_fraction.eq(.01))]
               .groupby(["source_arm", "calibration_mode"], as_index=False)
               .agg(outer_eras=("test_era", "nunique"), mean_top1_net_bps=("net_bps", "mean"), median_top1_net_bps=("net_bps", "median"),
                    worst_top1_net_bps=("net_bps", "min"), positive_eras=("net_bps", lambda s: int((s > 0).sum()))))
    # Pooled ranking is deliberately calculated only after each era's causal
    # map has put the source stack on the same bps scale.
    pooled_rows: list[dict[str, Any]] = []
    for (arm, mode), q in pp.groupby(["source_arm", "calibration_mode"], sort=True):
        pooled_rows += _rank_metrics(q, q.calibrated_bps.to_numpy(float), {"source_arm": arm, "calibration_mode": mode, "test_era": "ALL_OUTER", "prior_outer_eras": "rolling", "prior_rows": np.nan}, scope="pooled_global", period="ALL_OUTER")
    pooled = pd.DataFrame(pooled_rows)
    args.out.mkdir(parents=True)
    mm.to_parquet(args.out / "metrics.parquet", index=False); pp.to_parquet(args.out / "predictions.parquet", index=False)
    aa.to_parquet(args.out / "calibration_audit.parquet", index=False); summary.to_parquet(args.out / "summary.parquet", index=False); pooled.to_parquet(args.out / "pooled_global_metrics.parquet", index=False)
    lines = ["# Round E: causal shared-bps calibration", "", "D0/C2 control and D3 shadow forecasts are replayed unchanged. Each outer era uses only earlier out-of-era predictions whose exact H12 outcomes resolved before its first decision. E0/E1/E2 are additive common-bps calibration; E3 is the strongly-shrunk hierarchical affine challenger. None is an expert or gate.", "", "| Stack | Map | Mean outer top-1% net | Worst outer top-1% | Positive eras |", "|---|---|---:|---:|---:|"]
    for r in summary.sort_values(["source_arm", "calibration_mode"]).itertuples(index=False):
        lines.append(f"| {r.source_arm} | {r.calibration_mode} | {r.mean_top1_net_bps:+.2f} | {r.worst_top1_net_bps:+.2f} | {r.positive_eras}/{r.outer_eras} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {"schema": "tp6_shared_residual_round_e_calibration_v1", "status": "COMPLETED_DIAGNOSTIC_NO_PROMOTION", "source": str(D_ROOT), "geometry": "TP6/SL4/H12", "cost_bps": 100., "source_stacks": list(SOURCE_ARMS), "maps": list(MODES), "ranking": "global top-k after causal common-bps mapping", "no_local_or_regime_experts": True, "no_policy_or_admission_gate": True, "calibration": {"prior_oof_only": True, "resolution_contract": "exact label_available_ts; signal-close +1h entry +H12 (=13h fallback)", "side_shrink_rows": SIDE_SHRINK, "regime_shrink_rows": REGIME_SHRINK, "regime_weight_cap": REGIME_CAP, "identity_first_outer_era": True}, "selection": "diagnostic only; no arm advances without worst-era and pooled evidence"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.sort_values(["source_arm", "calibration_mode"]).to_string(index=False))


if __name__ == "__main__":
    main()
