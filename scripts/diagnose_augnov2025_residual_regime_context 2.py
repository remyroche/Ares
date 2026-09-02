#!/usr/bin/env python3
"""Diagnose frozen-July base/residual behavior on the Aug--Nov 2025 OOS bridge.

This is deliberately a *diagnostic*, not a model-selection or promotion
runner.  It consumes the sealed frozen-July score/economics ledger plus the
authoritative hourly sidecars.  All candidate, score, target and assessment
rows are one-hour decisions; the exact one-minute replay is already reduced
to the execution label on each hourly candidate.  No 2026 file is read.

The report keeps current-regime (BOCPD state) and transition (LGBM/BOCPD
onset) fields separate.  It quantifies base-to-residual top-10 replacement,
distribution/covariance movement, and simple pre-2026 conditional diagnostic
interactions.  It does not fit a context calibrator or use this OOS bridge to
choose one.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "data_perp/artifacts/augnov2025_common30_frozen_july_base_residual_oos_bridge_20260730_v1"
SIDECAR = ROOT / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1"
OUT = ROOT / "data_perp/artifacts/augnov2025_frozen_july_residual_regime_diagnosis_20260730_v2"
TOP = 0.10
FEATURE_CAP = 12

REGIME = [
    "bocpd__change_probability_mean", "bocpd__change_probability_max",
    "bocpd__run_length_mean", "bocpd__run_length_q05",
    "bocpd__run_length_entropy", "bocpd__signal_count",
    "bocpd__state_age_hours", "bocpd__is_persistent_24h",
    "bocpd__is_persistent_72h",
]
TRANSITION = [
    "lgbm_transition_probability", "lgbm_entropy", "lgbm_margin",
    "bocpd_onset_h1_probability", "bocpd_onset_h3_probability",
    "bocpd_onset_h6_probability", "bocpd_onset_h12_probability",
    "bocpd_stable_vs_transition_probability",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj: object) -> None:
    partial = path.with_name("." + path.name + ".partial")
    partial.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(partial, path)


def rank_ic(x: pd.DataFrame, score: str) -> float:
    return float(x[score].corr(x.execution_net_ev_12h, method="spearman"))


def select_global(x: pd.DataFrame, score: str) -> pd.Series:
    # Candidate id makes ties deterministic and ensures this is one global book,
    # never one top-k book per timestamp.
    ordered = x.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    return pd.Series(ordered.index[: math.ceil(len(x) * TOP)], index=ordered.index[: math.ceil(len(x) * TOP)])


def metrics(x: pd.DataFrame, score: str, selected: set[int], group: str) -> list[dict]:
    rows: list[dict] = []
    keys = [("aggregate", pd.Series("all", index=x.index)),
            ("month", x.__ts__.dt.strftime("%Y-%m")),
            ("week", x.__ts__.dt.strftime("%G-W%V")),
            ("side", x.side_name)]
    for scope, key in keys:
        for period, z in x.groupby(key, observed=True, sort=True):
            s = z.loc[z.index.isin(selected)]
            rows.append({"score": score, "scope": scope, "period": str(period),
                         "candidate_rows": int(len(z)), "selected_rows": int(len(s)),
                         "rank_ic": rank_ic(z, score),
                         "global_top10_net_ev": float(s.execution_net_ev_12h.mean()) if len(s) else np.nan,
                         "global_top10_gross_ev": float(s.execution_gross_ev_12h.mean()) if len(s) else np.nan,
                         "global_top10_cost": float(s.execution_cost_return.mean()) if len(s) else np.nan,
                         "global_top10_hit_rate": float(s.execution_net_ev_12h.gt(0).mean()) if len(s) else np.nan,
                         "group": group})
    return rows


def attach_features(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach the frozen base feature matrices by exact side/month row order."""
    result = x.copy()
    cols: list[str] = []
    for side in ("long", "short"):
        for month in range(8, 12):
            mask = result.side_name.eq(side) & result.__ts__.dt.month.eq(month)
            ix = result.index[mask]
            p = BRIDGE / side / f"month_2025_{month:02d}" / "base_oos_features.parquet"
            f = pd.read_parquet(p).reset_index(drop=True)
            if len(f) != len(ix):
                raise RuntimeError(f"feature row mismatch {side} {month}: {len(f)} != {len(ix)}")
            # The materializer created the matrix from the already stably sorted
            # score candidate frame.  Assert the score frame ordering before
            # positional attachment to avoid silently inventing a key.
            expected = result.loc[ix].sort_values(["__ts__", "candidate_id"], kind="stable").index
            if not expected.equals(ix):
                raise RuntimeError(f"unexpected bridge ordering {side} {month}")
            for c in f.columns:
                name = "base_feature__" + c
                result.loc[ix, name] = f[c].to_numpy()
                if name not in cols:
                    cols.append(name)
    return result, cols


def covariance_shift(x: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Feature covariance movement: Aug/Sep vs Oct/Nov, separate by side."""
    out = []
    for side, z in x.groupby("side_name", observed=True):
        early = z[z.__ts__.dt.month.isin([8, 9])]
        late = z[z.__ts__.dt.month.isin([10, 11])]
        usable = [c for c in fields if c in z and z[c].notna().all()]
        if len(usable) < 2:
            continue
        a, b = early[usable].corr(method="spearman"), late[usable].corr(method="spearman")
        for i, one in enumerate(usable):
            for two in usable[i + 1:]:
                out.append({"side_name": side, "field_a": one, "field_b": two,
                            "early_spearman": float(a.loc[one, two]), "late_spearman": float(b.loc[one, two]),
                            "delta_late_minus_early": float(b.loc[one, two] - a.loc[one, two]),
                            "abs_delta": float(abs(b.loc[one, two] - a.loc[one, two]))})
    return pd.DataFrame(out).sort_values("abs_delta", ascending=False, kind="stable")


def distribution_shift(x: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for side, z in x.groupby("side_name", observed=True):
        early, late = z[z.__ts__.dt.month.isin([8, 9])], z[z.__ts__.dt.month.isin([10, 11])]
        for c in fields:
            if c not in z or not z[c].notna().all():
                continue
            sd = float(z[c].std(ddof=0))
            delta = float(late[c].mean() - early[c].mean())
            rows.append({"side_name": side, "field": c, "early_mean": float(early[c].mean()),
                         "late_mean": float(late[c].mean()), "early_median": float(early[c].median()),
                         "late_median": float(late[c].median()), "pooled_sd": sd,
                         "standardized_mean_shift": delta / sd if sd > 0 else np.nan,
                         "abs_standardized_mean_shift": abs(delta / sd) if sd > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("abs_standardized_mean_shift", ascending=False, kind="stable")


def conditional_interactions(x: pd.DataFrame, fields: list[str], family: str) -> pd.DataFrame:
    """Univariate, fixed diagnostics of when residual correction aligns/hurts.

    The residual target proxy is actual net EV minus the frozen base EV mapping.
    We report the Spearman alignment of the residual delta with this proxy above
    and below the pre-2026 pooled median, separately by side and early/late
    periods.  This is intentionally discovery only, not a fitted gate.
    """
    rows = []
    x = x.copy()
    x["residual_target_proxy"] = x.execution_net_ev_12h - x.base_expected_ev
    for side, z in x.groupby("side_name", observed=True):
        for field in fields:
            if field not in z or not z[field].notna().all() or z[field].nunique() < 2:
                continue
            cut = float(z[field].median())
            for era, e in [("early_aug_sep", z[z.__ts__.dt.month.isin([8, 9])]),
                           ("late_oct_nov", z[z.__ts__.dt.month.isin([10, 11])])]:
                for bucket, b in [("low", e[e[field] <= cut]), ("high", e[e[field] > cut])]:
                    rows.append({"family": family, "side_name": side, "field": field, "era": era,
                                 "bucket": bucket, "cutoff_full_pre2026": cut, "rows": len(b),
                                 "residual_delta_target_proxy_ic": float(b.residual_delta_ev.corr(b.residual_target_proxy, method="spearman")),
                                 "residual_delta_net_ev_ic": float(b.residual_delta_ev.corr(b.execution_net_ev_12h, method="spearman")),
                                 "mean_residual_target_proxy": float(b.residual_target_proxy.mean())})
    q = pd.DataFrame(rows)
    if q.empty:
        return q
    wide = q.pivot_table(index=["family", "side_name", "field", "era"], columns="bucket",
                         values="residual_delta_target_proxy_ic", aggfunc="first").reset_index()
    wide["high_minus_low_alignment"] = wide.get("high") - wide.get("low")
    return q.merge(wide[["family", "side_name", "field", "era", "high_minus_low_alignment"]],
                   on=["family", "side_name", "field", "era"], how="left").sort_values(
                       "high_minus_low_alignment", key=lambda s: s.abs(), ascending=False, kind="stable")


def replacement(x: pd.DataFrame, base_selected: set[int], residual_selected: set[int]) -> pd.DataFrame:
    rows = []
    for scope, key in [("aggregate", pd.Series("all", index=x.index)),
                       ("month", x.__ts__.dt.strftime("%Y-%m")),
                       ("week", x.__ts__.dt.strftime("%G-W%V")),
                       ("side", x.side_name)]:
        for period, z in x.groupby(key, observed=True, sort=True):
            b, r = set(z.index) & base_selected, set(z.index) & residual_selected
            entered, exited = r - b, b - r
            # Exact accounting: residual book minus base book is only entry
            # economics less exit economics; shared candidates cancel.
            be, re = z.loc[list(b)], z.loc[list(r)]
            en, ex = z.loc[list(entered)], z.loc[list(exited)]
            rows.append({"scope": scope, "period": str(period), "base_selected": len(b),
                         "residual_selected": len(r), "shared": len(b & r), "entered": len(entered), "exited": len(exited),
                         "base_top10_net_ev": float(be.execution_net_ev_12h.mean()) if len(be) else np.nan,
                         "residual_top10_net_ev": float(re.execution_net_ev_12h.mean()) if len(re) else np.nan,
                         "residual_minus_base_net_ev": float(re.execution_net_ev_12h.mean() - be.execution_net_ev_12h.mean()) if len(be) and len(re) else np.nan,
                         "entered_net_ev": float(en.execution_net_ev_12h.mean()) if len(en) else np.nan,
                         "exited_net_ev": float(ex.execution_net_ev_12h.mean()) if len(ex) else np.nan,
                         "replacement_net_ev_gap": float(en.execution_net_ev_12h.mean() - ex.execution_net_ev_12h.mean()) if len(en) and len(ex) else np.nan,
                         "entered_residual_delta": float(en.residual_delta_ev.mean()) if len(en) else np.nan,
                         "exited_residual_delta": float(ex.residual_delta_ev.mean()) if len(ex) else np.nan})
    return pd.DataFrame(rows)


def seal(output: Path = OUT) -> Path:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite sealed output: {output}")
    manifest = json.loads((BRIDGE / "manifest.json").read_text())
    if manifest.get("status") != "SEALED_COMMON30_FROZEN_JULY_OOS_SCORE_BRIDGE_NON_PROMOTION":
        raise RuntimeError("unsealed bridge")
    if (BRIDGE / "manifest.sha256").read_text().split()[0] != sha(BRIDGE / "manifest.json"):
        raise RuntimeError("bridge manifest hash mismatch")
    x = pd.read_parquet(BRIDGE / "oos_predictions.parquet")
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    if len(x) != 175_680 or x.candidate_id.duplicated().any() or not (x.__ts__.astype("int64") % pd.Timedelta(hours=1).value == 0).all():
        raise RuntimeError("invalid hourly bridge")
    reg = pd.read_parquet(SIDECAR / "soft_regime_hourly.parquet").rename(columns={"source_utc": "__ts__"})
    trn = pd.read_parquet(SIDECAR / "soft_transition_hourly.parquet").rename(columns={"source_utc": "__ts__"})
    reg["__ts__"], trn["__ts__"] = pd.to_datetime(reg["__ts__"], utc=True), pd.to_datetime(trn["__ts__"], utc=True)
    if reg.__ts__.duplicated().any() or trn.__ts__.duplicated().any():
        raise RuntimeError("sidecar duplicate hourly state")
    x = x.merge(reg[["__ts__", *REGIME]], on="__ts__", how="left", validate="many_to_one")
    x = x.merge(trn[["__ts__", *TRANSITION]], on="__ts__", how="left", validate="many_to_one")
    if x[[*REGIME, *TRANSITION]].isna().any().any():
        raise RuntimeError("sidecar coverage failure")
    x, base_features = attach_features(x)
    # The long contract has 31 fields and the short one eight.  A long-only
    # field is necessarily null on the short rows (and vice versa), so retain
    # all fields with any support.  The side-local helpers below then require
    # complete support within the applicable side before measuring it.
    base_features = [c for c in base_features if x[c].notna().any()]
    fields = [*REGIME, *TRANSITION, *base_features]
    base_selected = set(select_global(x, "score_base_alpha").index)
    residual_selected = set(select_global(x, "score_residual_expected_ev").index)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix="." + output.name + "."))
    try:
        rows = metrics(x, "score_base_alpha", base_selected, "base") + metrics(x, "score_residual_expected_ev", residual_selected, "residual")
        summary = pd.DataFrame(rows)
        repl = replacement(x, base_selected, residual_selected)
        dist = distribution_shift(x, fields)
        cov = covariance_shift(x, fields)
        regime_inter = conditional_interactions(x, REGIME, "regime")
        transition_inter = conditional_interactions(x, TRANSITION, "transition")
        summary.to_csv(stage / "score_metrics_month_week_side.csv", index=False)
        repl.to_csv(stage / "residual_replacement_attribution.csv", index=False)
        dist.to_csv(stage / "feature_context_distribution_shift.csv", index=False)
        cov.to_csv(stage / "feature_context_covariance_shift.csv", index=False)
        regime_inter.to_csv(stage / "regime_conditional_residual_alignment.csv", index=False)
        transition_inter.to_csv(stage / "transition_conditional_residual_alignment.csv", index=False)
        pd.DataFrame({"candidate_id": x.candidate_id, "__ts__": x.__ts__, "side_name": x.side_name,
                      "selected_base_global_top10": x.index.isin(base_selected),
                      "selected_residual_global_top10": x.index.isin(residual_selected),
                      "residual_delta_ev": x.residual_delta_ev,
                      "execution_net_ev_12h": x.execution_net_ev_12h}).to_parquet(stage / "selection_membership.parquet", index=False)
        checks = {"rows": int(len(x)), "unique_candidates": int(x.candidate_id.nunique()),
                  "all_hourly": bool((x.__ts__.astype("int64") % pd.Timedelta(hours=1).value == 0).all()),
                  "months": x.__ts__.dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
                  "no_2026_rows": bool(x.__ts__.dt.year.lt(2026).all()),
                  "regime_coverage": float(x[REGIME].notna().all(axis=1).mean()),
                  "transition_coverage": float(x[TRANSITION].notna().all(axis=1).mean()),
                  "base_global_top10_rows": len(base_selected), "residual_global_top10_rows": len(residual_selected),
                  "base_feature_fields_attached": len(base_features)}
        dump(stage / "validation.json", checks)
        files = sorted(p for p in stage.iterdir() if p.is_file())
        out_manifest = {"schema": "augnov2025_frozen_july_residual_regime_diagnosis_v2",
                        "status": "SEALED_OOS_DIAGNOSTIC_NON_PROMOTION",
                        "promotion_eligible": False,
                        "scope": "Aug-November 2025 common-30 frozen-July base/residual score bridge; no 2026 outcomes",
                        "decision_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only",
                        "selection": "one global top10% across all 175680 hourly candidates per score, candidate-id deterministic tie-break",
                        "regime_transition_separation": "regime uses BOCPD state fields; transition uses LGBM/BOCPD onset fields; no combined interaction score is fitted",
                        "inputs_sha256": {str((BRIDGE / "manifest.json").resolve()): sha(BRIDGE / "manifest.json"),
                                          str((BRIDGE / "oos_predictions.parquet").resolve()): sha(BRIDGE / "oos_predictions.parquet"),
                                          str((SIDECAR / "manifest.json").resolve()): sha(SIDECAR / "manifest.json")},
                        "validation": checks,
                        "outputs_sha256": {p.name: sha(p) for p in files}}
        dump(stage / "manifest.json", out_manifest)
        (stage / "manifest.sha256").write_text(f"{sha(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
        return output
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(seal())
