#!/usr/bin/env python3
"""Retrospective 2024--26 comparison of the frozen short O/C 2x2 contracts.

This is deliberately a *research* producer.  The four O/C contracts were
pre-registered for an untouched August-2026 evaluation.  The user has now
requested a broad historical comparison, so this runner never updates that
registry and writes a separate, explicitly non-untouched artifact.

Every arm uses the same:

    P0 target-free candidates
    -> O250/H6, uniform LightGBM + Platt
    -> C3 normalized-regret conditional LightGBM
    -> strict-prequential K0 (Platt, isotonic mu1, anchor5 mu0/k=500)
    -> absolute expected-policy-net admission >= 75 bps

Only the frozen O45/O30 and C59/C40 feature contracts differ.  All fitting
and mapping inputs obey ``label_available_at < held_month_start``.  Invalid
paths are scored but excluded from supervised fitting and realised economics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preregister_strict_r3_short_p0_oc_k0_untouched_2x2 as prereg  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round4_k0_refinement as r4  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_retrospective_2x2_v1"
REGISTRY = prereg.OUT / "untouched_2x2_registry.json"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_retrospective_2x2_202408_202607_20260822_v2"
START = pd.Timestamp("2024-05-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
K0_MU1 = ("isotonic", 0)
K0_MU0 = ("anchor5", 500)
K0_ADMISSION = ("absolute", 75.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: pd.Series | np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(pd.to_numeric(value, errors="coerce"), dtype=float), nan=np.nan, posinf=np.nan, neginf=np.nan)


def _safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    x, y = _finite(left), _finite(right)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return float("nan")
    return float(pd.Series(x[valid]).corr(pd.Series(y[valid]), method="spearman"))


def _events(frame: pd.DataFrame) -> np.ndarray:
    return r1._event(frame, r3.SPEC).astype(bool)


def _valid(frame: pd.DataFrame) -> pd.Series:
    return r1._valid_label(frame) & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()


def _conversion_metrics(part: pd.DataFrame) -> dict[str, float | int]:
    valid = part.loc[_valid(part) & _events(part)].copy()
    if valid.empty:
        return {
            "conversion_rows": 0, "conversion_net_rank_ic": float("nan"),
            "conversion_populated_bins": 0, "conversion_monotonic_violations": 0,
        }
    score = pd.to_numeric(valid["conversion_score"], errors="coerce")
    net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    ranks = score.rank(method="first", pct=True)
    bins = pd.cut(ranks, np.linspace(0.0, 1.0, 11), include_lowest=True, duplicates="drop")
    means = valid.assign(_bin=bins).groupby("_bin", observed=True)["policy_net_bps"].mean().to_numpy(float)
    return {
        "conversion_rows": int(len(valid)),
        "conversion_net_rank_ic": _safe_spearman(score, net),
        "conversion_populated_bins": int(len(means)),
        "conversion_monotonic_violations": int(np.sum(np.diff(means) < 0.0)) if len(means) > 1 else 0,
    }


def _month_metrics(prediction: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k0_rows: list[dict[str, Any]] = []
    o_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    for month, part in prediction.groupby("held_month", sort=True):
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        working = part.copy()
        # r1's O diagnostics deliberately consume the final, prequentially
        # calibrated probability rather than the provisional upstream one.
        working["opportunity_probability"] = working["opportunity_probability_round4"]
        k0 = r1._k0_metrics(working, r3.SPEC, month_start)
        k0["arm"] = arm
        k0_rows.append(k0)
        o = r1._o_metrics(working, r3.SPEC, month_start)
        o["arm"] = arm
        o_rows.append(o)
        c_rows.append({"arm": arm, "held_month": month, **_conversion_metrics(working)})
    return pd.DataFrame(k0_rows), pd.DataFrame(o_rows), pd.DataFrame(c_rows)


def _period_metrics(prediction: pd.DataFrame, monthly: pd.DataFrame, opportunity: pd.DataFrame, conversion: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, mask in {
        "2024": prediction["held_month"].str.startswith("2024"),
        "2025": prediction["held_month"].str.startswith("2025"),
        "2026": prediction["held_month"].str.startswith("2026"),
        "2024_2026_available": pd.Series(True, index=prediction.index),
    }.items():
        part = prediction.loc[mask].copy()
        selected = part.loc[pd.to_numeric(part["K0_expected_policy_net_bps"], errors="coerce").ge(75.0)]
        valid = selected.loc[_valid(selected)]
        net = _finite(valid["policy_net_bps"])
        net = net[np.isfinite(net)]
        months = monthly.loc[monthly["held_month"].str[:4].eq(period[:4])] if period != "2024_2026_available" else monthly
        o = opportunity.loc[opportunity["held_month"].str[:4].eq(period[:4])] if period != "2024_2026_available" else opportunity
        c = conversion.loc[conversion["held_month"].str[:4].eq(period[:4])] if period != "2024_2026_available" else conversion
        rows.append({
            "arm": arm, "period": period, "months": int(len(months)),
            "scored_candidates": int(len(part)), "selected_candidates": int(len(selected)),
            "outcome_known_candidates": int(len(valid)),
            "outcome_coverage": float(len(valid) / len(selected)) if len(selected) else float("nan"),
            "admission_rate": float(len(selected) / len(part)) if len(part) else float("nan"),
            "net_bps_per_trade": float(np.mean(net)) if len(net) else float("nan"),
            "total_net_bps": float(np.sum(net)) if len(net) else 0.0,
            "cvar10_bps": r1._cvar(net),
            "positive_fraction": float(np.mean(net > 0.0)) if len(net) else float("nan"),
            "fraction_lt_neg200": float(np.mean(net < -200.0)) if len(net) else float("nan"),
            "worst_month_net_bps_per_trade": float(months["net_bps_per_trade"].min()) if len(months) else float("nan"),
            "positive_months": int((months["net_bps_per_trade"] > 0.0).sum()) if len(months) else 0,
            "o_auc_mean": float(o["auc"].mean()) if len(o) else float("nan"),
            "o_prauc_mean": float(o["prauc"].mean()) if len(o) else float("nan"),
            "o_brier_mean": float(o["brier"].mean()) if len(o) else float("nan"),
            "o_lift20_mean": float(o["lift_top20"].mean()) if len(o) else float("nan"),
            "c_net_rank_ic_mean": float(c["conversion_net_rank_ic"].mean()) if len(c) else float("nan"),
            "c_monotonic_violations": int(c["conversion_monotonic_violations"].sum()) if len(c) else 0,
        })
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda x: f"{x:.3f}" if isinstance(x, float) and np.isfinite(x) else ("—" if isinstance(x, float) else x))
    try:
        return view.to_markdown(index=False)
    except ImportError:
        # ``tabulate`` is optional in the Ares runtime.  Reports must not make
        # a completed immutable model run depend on that presentation package.
        columns = [str(column) for column in view.columns]
        rows = [[str(value).replace("|", "\\|") for value in row] for row in view.itertuples(index=False, name=None)]
        return "\n".join((
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ))


def _assert_prequential(prediction: pd.DataFrame, audit: pd.DataFrame, arm: str) -> None:
    complete = audit.loc[audit["status"].eq("complete")].copy()
    if complete.empty:
        raise AssertionError(f"{arm}: no completed strict-prequential O/C folds")
    starts = pd.to_datetime(complete["held_month"] + "-01", utc=True)
    if (pd.to_numeric(complete["outer_train_rows"], errors="coerce") < r1.MIN_OUTER_TRAIN_ROWS).any():
        raise AssertionError(f"{arm}: invalid outer-training support")
    for held_month, start in zip(complete["held_month"], starts, strict=True):
        history = prediction.loc[
            prediction["__decision_ts__"].lt(start) & prediction["__label_available_at__"].lt(start) & _valid(prediction)
        ]
        if len(history) and not history["__label_available_at__"].lt(start).all():
            raise AssertionError(f"{arm}/{held_month}: mapping history contains unresolved label")


def run(out: Path = OUT) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    registry = json.loads(REGISTRY.read_text())
    if registry["status"] != "pre_registered_not_evaluated":
        raise AssertionError("unexpected preregistration registry state")
    stack = registry["frozen_stack"]
    if stack["K0"]["admission"]["threshold_bps"] != 75.0:
        raise AssertionError("frozen 2x2 K0 threshold changed")
    frame, _, _, source_hashes = r3._load_frame()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["__label_available_at__"] = pd.to_datetime(frame["__label_available_at__"], utc=True)
    frame = frame.loc[frame["__decision_ts__"].ge(START) & frame["__decision_ts__"].lt(END)].copy()
    if frame.empty:
        raise AssertionError("no historical candidate population")
    rows: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    monthly_all: list[pd.DataFrame] = []
    opportunity_all: list[pd.DataFrame] = []
    conversion_all: list[pd.DataFrame] = []
    period_all: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    out.mkdir(parents=True)
    for arm in registry["arms"]:
        name = str(arm["arm"])
        raw, outer_audit = r3._run_target(
            frame,
            tuple(arm["opportunity_features"]),
            tuple(arm["conversion_features"]),
            r3b.TARGET,
            # Seeds are frozen across all four arms.  Feature-contract
            # differences must not be confounded with LightGBM randomness.
            r3b.C_SEED,
            "uniform",
            o_seed=r3b.O_SEED,
        )
        anchors = frame.loc[:, ["candidate_id", r4.P0_ANCHOR]]
        raw = raw.merge(anchors, on="candidate_id", how="left", validate="one_to_one")
        if raw[r4.P0_ANCHOR].isna().any():
            raise AssertionError(f"{name}: P0 anchors missing")
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True)
        raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
        mapped, k0_audit = r4._replay(raw, mu1=K0_MU1, mu0=K0_MU0, admission=K0_ADMISSION)
        _assert_prequential(mapped, outer_audit, name)
        if not pd.to_numeric(mapped["K0_train_p80_expected_policy_net_bps"], errors="coerce").eq(75.0).all():
            raise AssertionError(f"{name}: K0 admission is not the frozen absolute 75-bps threshold")
        monthly, opportunity, conversion = _month_metrics(mapped, name)
        periods = _period_metrics(mapped, monthly, opportunity, conversion, name)
        mapped["arm"] = name
        mapped.to_parquet(out / f"{name}_strict_prequential_predictions.parquet", index=False, compression="zstd")
        outer_audit.assign(arm=name).to_parquet(out / f"{name}_outer_training_audit.parquet", index=False, compression="zstd")
        k0_audit.assign(arm=name).to_parquet(out / f"{name}_k0_mapping_audit.parquet", index=False, compression="zstd")
        rows.append(mapped)
        audits.extend((outer_audit.assign(arm=name, stage="O_C"), k0_audit.assign(arm=name, stage="K0")))
        monthly_all.append(monthly); opportunity_all.append(opportunity); conversion_all.append(conversion); period_all.append(periods)
        all_period = periods.loc[periods["period"].eq("2024_2026_available")].iloc[0]
        summaries.append({
            "arm": name, "description": arm["description"],
            "opportunity_feature_count": arm["opportunity_feature_count"],
            "conversion_feature_count": arm["conversion_feature_count"],
            "net_bps_per_trade": all_period["net_bps_per_trade"], "total_net_bps": all_period["total_net_bps"],
            "outcome_known_candidates": all_period["outcome_known_candidates"],
            "worst_month_net_bps_per_trade": all_period["worst_month_net_bps_per_trade"],
            "cvar10_bps": all_period["cvar10_bps"], "o_auc_mean": all_period["o_auc_mean"],
            "o_lift20_mean": all_period["o_lift20_mean"], "c_net_rank_ic_mean": all_period["c_net_rank_ic_mean"],
        })
    comparison = pd.DataFrame(summaries).sort_values("arm", kind="stable").reset_index(drop=True)
    base = comparison.loc[comparison["arm"].eq("A0")].iloc[0]
    for column in ("net_bps_per_trade", "total_net_bps", "worst_month_net_bps_per_trade", "cvar10_bps", "o_auc_mean", "o_lift20_mean", "c_net_rank_ic_mean"):
        comparison[f"delta_vs_A0_{column}"] = comparison[column] - float(base[column])
    monthly_frame = pd.concat(monthly_all, ignore_index=True)
    opportunity_frame = pd.concat(opportunity_all, ignore_index=True)
    conversion_frame = pd.concat(conversion_all, ignore_index=True)
    period_frame = pd.concat(period_all, ignore_index=True)
    pd.concat(rows, ignore_index=True).loc[:, ["arm", "candidate_id", "__decision_ts__", "__label_available_at__", "held_month", "opportunity_raw_score", "opportunity_probability_round4", "conversion_score", "K0_expected_policy_net_bps", "K0_train_p80_expected_policy_net_bps", "policy_net_bps", "rich_path_label_valid", "rich_path_target_invalid", "policy_path_valid"]].to_parquet(out / "all_arms_score_ledger.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "strict_prequential_audit.parquet", index=False, compression="zstd")
    monthly_frame.to_parquet(out / "monthly_admission_metrics.parquet", index=False, compression="zstd")
    opportunity_frame.to_parquet(out / "monthly_opportunity_metrics.parquet", index=False, compression="zstd")
    conversion_frame.to_parquet(out / "monthly_conversion_metrics.parquet", index=False, compression="zstd")
    period_frame.to_parquet(out / "period_metrics.parquet", index=False, compression="zstd")
    comparison.to_parquet(out / "comparison_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete_retroactive_research_not_untouched", "side": "short",
        "scope": "May 2024 through July 2026 strict-prequential retrospective comparison requested after 2x2 preregistration; not eligible as untouched promotion evidence",
        "period": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "arms": registry["arms"], "frozen_stack": stack,
        "K0": {"mu1": list(K0_MU1), "mu0": list(K0_MU0), "admission": list(K0_ADMISSION)},
        "causality": {
            "outer_O_C": "label_available_at < held month start",
            "K0": "only prior outer-OOS scores with labels resolved before held month",
            "candidate_handling": "target-free candidates scored before invalid outcomes excluded",
            "portfolio": "not included: this is the short P0/O/C/K0 admission stack, not a side-aware portfolio simulation",
        },
        "inputs": {"preregistration_sha256": _sha256(REGISTRY), **source_hashes},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short P0 → O → C → K0: retrospective 2×2 comparison", "",
        "This is retrospective research evidence for the previously frozen A0–A3 contracts. It intentionally does **not** modify their untouched-forward registry or provide promotion evidence.", "",
        "## Aggregate causal admission economics", "",
        _markdown(comparison, ["arm", "description", "opportunity_feature_count", "conversion_feature_count", "outcome_known_candidates", "net_bps_per_trade", "delta_vs_A0_net_bps_per_trade", "total_net_bps", "delta_vs_A0_total_net_bps", "worst_month_net_bps_per_trade", "cvar10_bps"]), "",
        "## Yearly causal admission economics", "",
        _markdown(period_frame, ["arm", "period", "months", "scored_candidates", "selected_candidates", "outcome_known_candidates", "admission_rate", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "worst_month_net_bps_per_trade", "positive_months"]), "",
        "## Opportunity and conversion diagnostics", "",
        _markdown(period_frame, ["arm", "period", "o_auc_mean", "o_prauc_mean", "o_brier_mean", "o_lift20_mean", "c_net_rank_ic_mean", "c_monotonic_violations"]), "",
        "## Interpretation constraints", "",
        "- Metrics use only admissions from the frozen absolute 75-bps K0 expected-policy-net threshold, with invalid outcomes excluded after scoring.",
        "- No portfolio auction, execution slippage, or live readiness claim is made in this short-side research artifact.",
        "- 2024 warm-up months may be absent where strict prior-resolved O/C/K0 support is insufficient; audit rows explicitly record this.",
        "- Since 2024–2026 outcomes are now consumed for this comparison, the original A0–A3 untouched protocol remains preserved but cannot treat these years as fresh validation evidence.", "",
    ]
    (out / "SHORT_P0_OC_K0_RETROSPECTIVE_2X2_REPORT.md").write_text("\n".join(report))
    return out


def finalize_existing(out: Path) -> Path:
    """Render the report/manifest for a completed ledger without refitting.

    This recovery path exists so a report-formatting dependency failure cannot
    force a costly retrain or overwrite the immutable score artifacts.
    """
    required = (
        "comparison_metrics.parquet", "period_metrics.parquet",
        "monthly_admission_metrics.parquet", "monthly_opportunity_metrics.parquet",
        "monthly_conversion_metrics.parquet", "strict_prequential_audit.parquet",
        *(f"{arm}_strict_prequential_predictions.parquet" for arm in ("A0", "A1", "A2", "A3")),
    )
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalise incomplete comparison: {missing}")
    if (out / "SHORT_P0_OC_K0_RETROSPECTIVE_2X2_REPORT.md").exists():
        raise FileExistsError(f"comparison report is already finalised: {out}")
    registry = json.loads(REGISTRY.read_text())
    comparison = pd.read_parquet(out / "comparison_metrics.parquet")
    period_frame = pd.read_parquet(out / "period_metrics.parquet")
    manifest = {
        "schema": SCHEMA, "status": "complete_retroactive_research_not_untouched", "side": "short",
        "scope": "May 2024 through July 2026 strict-prequential retrospective comparison requested after 2x2 preregistration; not eligible as untouched promotion evidence",
        "period": {"start": START.isoformat(), "end_exclusive": END.isoformat()},
        "arms": registry["arms"], "frozen_stack": registry["frozen_stack"],
        "K0": {"mu1": list(K0_MU1), "mu0": list(K0_MU0), "admission": list(K0_ADMISSION)},
        "causality": {
            "outer_O_C": "label_available_at < held month start",
            "K0": "only prior outer-OOS scores with labels resolved before held month",
            "candidate_handling": "target-free candidates scored before invalid outcomes excluded",
            "portfolio": "not included: this is the short P0/O/C/K0 admission stack, not a side-aware portfolio simulation",
        },
        "recovery": "report/manifest finalised from already-completed score/metric ledgers; no model or mapper refit",
        "inputs": {"preregistration_sha256": _sha256(REGISTRY)},
    }
    report = [
        "# Short P0 → O → C → K0: retrospective 2×2 comparison", "",
        "This is retrospective research evidence for the previously frozen A0–A3 contracts. It intentionally does **not** modify their untouched-forward registry or provide promotion evidence.", "",
        "## Aggregate causal admission economics", "",
        _markdown(comparison, ["arm", "description", "opportunity_feature_count", "conversion_feature_count", "outcome_known_candidates", "net_bps_per_trade", "delta_vs_A0_net_bps_per_trade", "total_net_bps", "delta_vs_A0_total_net_bps", "worst_month_net_bps_per_trade", "cvar10_bps"]), "",
        "## Yearly causal admission economics", "",
        _markdown(period_frame, ["arm", "period", "months", "scored_candidates", "selected_candidates", "outcome_known_candidates", "admission_rate", "net_bps_per_trade", "total_net_bps", "cvar10_bps", "worst_month_net_bps_per_trade", "positive_months"]), "",
        "## Opportunity and conversion diagnostics", "",
        _markdown(period_frame, ["arm", "period", "o_auc_mean", "o_prauc_mean", "o_brier_mean", "o_lift20_mean", "c_net_rank_ic_mean", "c_monotonic_violations"]), "",
        "## Interpretation constraints", "",
        "- Metrics use only admissions from the frozen absolute 75-bps K0 expected-policy-net threshold, with invalid outcomes excluded after scoring.",
        "- No portfolio auction, execution slippage, or live readiness claim is made in this short-side research artifact.",
        "- 2024 warm-up months may be absent where strict prior-resolved O/C/K0 support is insufficient; audit rows explicitly record this.",
        "- Since 2024–2026 outcomes are now consumed for this comparison, the original A0–A3 untouched protocol remains preserved but cannot treat these years as fresh validation evidence.", "",
    ]
    # The original run writes its manifest before rendering Markdown.  Preserve
    # that already-complete provenance record if the formatter alone failed.
    if not (out / "run_manifest.json").exists():
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "SHORT_P0_OC_K0_RETROSPECTIVE_2X2_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--finalize-existing", action="store_true", help="render report/manifest from a completed score ledger; never refit")
    args = parser.parse_args()
    print(finalize_existing(args.out) if args.finalize_existing else run(args.out))


if __name__ == "__main__":
    main()
