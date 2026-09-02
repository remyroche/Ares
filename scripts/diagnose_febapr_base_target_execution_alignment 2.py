#!/usr/bin/env python3
"""Diagnose frozen base-target rank quality versus exact execution payoff.

All score selection is one pooled global top-decile per calendar month after a
causal 21-day execution-EV map.  No timestamp-local selection is performed.
Transition labels are ex-post diagnostic strata only; v3 context fields are
pre-entry-only diagnostic strata and are never used to alter the score.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet"
POP = ROOT / "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v2/population.parquet"
CONTEXT = ROOT / "data_perp/artifacts/febapr2025_strict_residual_gross_regime_context_20260729_v3/panel.parquet"
OUT = ROOT / "data_perp/artifacts/febapr2025_base_target_execution_alignment_20260729_v1"
ID = ["candidate_id", "side_name", "__symbol__", "__ts__"]
CONTEXT_FEATURES = ["range_24h_pct", "__meta_raw__volatility_zscore", "trend_r2_24", "jump_intensity"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def top_global(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    """One deterministic pooled top 10%, never per timestamp."""
    n = max(1, int(np.ceil(len(frame) * 0.10)))
    return frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n).copy()


def causal_map(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Map frozen score to exact net EV using only prior resolved 21-day rows."""
    out = pd.Series(index=frame.index, dtype=float)
    rows = []
    for day in sorted(frame["__ts__"].dt.floor("D").unique()):
        mask = frame["__ts__"].dt.floor("D").eq(day)
        hist = frame.loc[
            frame["__ts__"].lt(day)
            & frame["execution_label_end_utc"].lt(day)
            & frame["__ts__"].ge(day - pd.Timedelta(days=21))
        ]
        if len(hist) < 300 or hist["base_oof_score"].nunique() < 2:
            out.loc[mask] = frame.loc[mask, "base_oof_score"]
            rows.append({"day": day, "rows": int(mask.sum()), "history_rows": int(len(hist)), "mode": "raw_fallback"})
        else:
            model = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(
                hist["base_oof_score"], hist["execution_net_ev_12h"]
            )
            out.loc[mask] = model.predict(frame.loc[mask, "base_oof_score"])
            rows.append({"day": day, "rows": int(mask.sum()), "history_rows": int(len(hist)), "mode": "causal_21d_isotonic"})
    return out, pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    target = frame["__first_touch_target_soft__"]
    return {
        "rows": int(len(frame)),
        "score_target_spearman": float(frame[[score, "__first_touch_target_soft__"]].corr(method="spearman").iloc[0, 1]),
        "score_gross_spearman": float(frame[[score, "execution_gross_ev_12h"]].corr(method="spearman").iloc[0, 1]),
        "score_cost_spearman": float(frame[[score, "execution_cost_return"]].corr(method="spearman").iloc[0, 1]),
        "score_net_spearman": float(frame[[score, "execution_net_ev_12h"]].corr(method="spearman").iloc[0, 1]),
        "target_gross_spearman": float(target.corr(frame["execution_gross_ev_12h"], method="spearman")),
        "target_cost_spearman": float(target.corr(frame["execution_cost_return"], method="spearman")),
        "target_net_spearman": float(target.corr(frame["execution_net_ev_12h"], method="spearman")),
    }


def payoff(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "selected_rows": int(len(frame)),
        "gross_bps": float(frame["execution_gross_ev_12h"].mean() * 1e4),
        "cost_bps": float(frame["execution_cost_return"].mean() * 1e4),
        "net_bps": float(frame["execution_net_ev_12h"].mean() * 1e4),
        "positive_net_rate": float(frame["execution_net_ev_12h"].gt(0).mean()),
        "native_target_mean": float(frame["__first_touch_target_soft__"].mean()),
    }


def selected_slice(frame: pd.DataFrame, score: str, fields: list[str]) -> pd.DataFrame:
    rows = []
    for month, group in frame.groupby("month", sort=True):
        selected = top_global(group, score)
        for values, part in selected.groupby(fields, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            rows.append({"score": score, "month": month, **dict(zip(fields, map(str, values))), **payoff(part)})
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    base = pd.read_parquet(BASE, columns=[*ID, "__decision_ts__", "__first_touch_target_soft__", "base_oof_score"])
    pop = pd.read_parquet(POP, columns=["candidate_id", "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "execution_exit_reason", "execution_exit_minute", "execution_mfe_return_12h", "execution_mae_return_12h", "transition_event_id", "expost_transition_active", "transition_window_member"])
    x = base.merge(pop, on="candidate_id", how="inner", validate="one_to_one")
    if len(x) != 509_868 or x["candidate_id"].duplicated().any():
        raise ValueError("base/exact execution identity contract fails")
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True)
    x["execution_label_end_utc"] = pd.to_datetime(x["execution_label_end_utc"], utc=True)
    x["month"] = x["__ts__"].dt.strftime("%Y-%m")
    if not np.allclose(x["execution_gross_ev_12h"] - x["execution_cost_return"], x["execution_net_ev_12h"], atol=1e-12, rtol=0):
        raise ValueError("exact execution accounting fails")
    x["mapped_execution_ev"] , mapping = causal_map(x)
    x["transition_phase"] = np.where(x["expost_transition_active"].eq(1), "active", np.where(x["transition_window_member"].eq(1), "window_nonactive", "outside"))

    context = pd.read_parquet(CONTEXT, columns=["candidate_id", *CONTEXT_FEATURES])
    x = x.merge(context, on="candidate_id", how="left", validate="one_to_one", indicator="context_join")
    x["has_context"] = x["context_join"].eq("both")
    x = x.drop(columns="context_join")
    if int(x["has_context"].sum()) != 140_682:
        raise ValueError("v3 strict context coverage mismatch")

    month_rows = []
    selected_rows = []
    overlap_rows = []
    for month, group in x.groupby("month", sort=True):
        raw = top_global(group, "base_oof_score")
        mapped = top_global(group, "mapped_execution_ev")
        for score, selected in (("base_oof_score", raw), ("mapped_execution_ev", mapped)):
            month_rows.append({"month": month, "score": score, **metrics(group, score), **payoff(selected)})
            for side, part in selected.groupby("side_name", sort=True):
                selected_rows.append({"month": month, "score": score, "side_name": side, **payoff(part)})
        overlap_rows.append({"month": month, "raw_mapped_top10_candidate_jaccard": float(len(set(raw.candidate_id) & set(mapped.candidate_id)) / len(set(raw.candidate_id) | set(mapped.candidate_id)))})
    month_metrics = pd.DataFrame(month_rows)
    selected_by_side = pd.DataFrame(selected_rows)
    stability = pd.DataFrame(overlap_rows)

    decile_rows = []
    for month, group in x.groupby("month", sort=True):
        ranks = group["base_oof_score"].rank(method="first", pct=True)
        local = group.assign(score_decile=np.minimum((ranks * 10).astype(int), 9) + 1)
        for decile, part in local.groupby("score_decile", sort=True):
            decile_rows.append({"month": month, "score_decile": int(decile), **payoff(part), "rows": int(len(part))})
    monotonicity = pd.DataFrame(decile_rows)

    exits = pd.concat([selected_slice(x, score, ["execution_exit_reason"]) for score in ("base_oof_score", "mapped_execution_ev")], ignore_index=True)
    transition = pd.concat([selected_slice(x, score, ["transition_phase"]) for score in ("base_oof_score", "mapped_execution_ev")], ignore_index=True)
    context_rows = []
    for month, group in x.loc[x.has_context].groupby("month", sort=True):
        for score in ("base_oof_score", "mapped_execution_ev"):
            selected = top_global(group, score)
            for feature in CONTEXT_FEATURES:
                bins = pd.qcut(group[feature], 3, labels=("low", "mid", "high"), duplicates="drop")
                selected_bins = bins.reindex(selected.index)
                for bucket, part in selected.assign(context_bucket=selected_bins.astype(str)).groupby("context_bucket", sort=True):
                    context_rows.append({"month": month, "score": score, "feature": feature, "context_bucket": bucket, **payoff(part)})
    context_strata = pd.DataFrame(context_rows)

    temp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    mapping.to_parquet(temp / "causal_mapping_coverage.parquet", index=False, compression="zstd")
    month_metrics.to_parquet(temp / "month_global_top10_alignment.parquet", index=False, compression="zstd")
    selected_by_side.to_parquet(temp / "top10_side_composition.parquet", index=False, compression="zstd")
    monotonicity.to_parquet(temp / "raw_score_decile_payoff.parquet", index=False, compression="zstd")
    exits.to_parquet(temp / "top10_exit_composition.parquet", index=False, compression="zstd")
    transition.to_parquet(temp / "top10_transition_diagnostic.parquet", index=False, compression="zstd")
    context_strata.to_parquet(temp / "strict_context_preentry_strata.parquet", index=False, compression="zstd")
    stability.to_parquet(temp / "raw_vs_mapped_rank_stability.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "febapr2025_base_target_execution_alignment_v1", "status": "DIAGNOSTIC_ONLY_FROZEN_OOF_EXACT_EXECUTION",
        "rows": int(len(x)), "context_rows": int(x.has_context.sum()),
        "selection": "one pooled global top 10% per calendar month after score mapping; never per timestamp",
        "mapping": "daily score->exact-net isotonic map fits only the prior resolved 21-day window; raw fallback before sufficient history",
        "causality": {"scores": "frozen base OOF", "exact_execution": "diagnostic payoff only", "transition_phase": "ex-post diagnostic only", "context": "v3 archived pre-entry fields only"},
        "sources_sha256": {str(BASE): sha(BASE), str(POP): sha(POP), str(CONTEXT): sha(CONTEXT)},
        "outputs_sha256": {p.name: sha(p) for p in sorted(temp.glob("*.parquet"))},
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (temp / "manifest.sha256").write_text(f"{sha(temp / 'manifest.json')}  manifest.json\n")
    os.replace(temp, OUT)


if __name__ == "__main__":
    main()
