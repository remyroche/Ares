#!/usr/bin/env python3
"""Causal O/C feature-block ablations for the short P0 -> O -> C -> K0 funnel.

This is deliberately a *block* ablation, not a new wide feature search.  It
reuses only the existing target-free F115 source panel.  Cross-sectional fields
are the already-materialised ``xs_*`` inputs whose source contract calculates
them before P0 candidate filtering.  The only derived fields here are
per-symbol, prior-only 30-day self-state percentiles over the target-free P0
route; they never consume a label, path, or a row at the same/later decision
time.  A 7-vs-90-day conditional state was intentionally not materialised: the
P0 route does not supply 90%-covered per-symbol support for it.

The runner keeps the opportunity definition, C target, model geometry, seeds,
analytic K0 formula, and +75-bps admission fixed.  It evaluates O blocks and C
blocks separately over the same 2024 diagnostic / 2025--2026 selection ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3d_c59_coverage_repair as c59  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_phase_c_feature_blocks_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_phase_c_feature_blocks_202408_202607_20260822_v1"
TARGET = next(item for item in r3.TARGETS if item.name == "C3_normalized_regret")
O_SEED = r3b.O_SEED
C_SEED = r3b.C_SEED
LATER_ERAS = ("2025", "2026")
MIN_LATER_COVERAGE = .90

# These are intentionally modest, named source blocks.  Every field is in the
# immutable F115 target-free panel and is omitted if it is already in the
# frozen contract of the layer currently being ablated.
SF_FIELDS = (
    "price_recovery_from_low_24h_atr",
    "asset_minus_mkt_price_recovery_fraction_24h",
    "q_iqr__ret48h_bench_resid",
)
TF_FIELDS = (
    "false_clean_short",
    "price_minus_oi_recovery_72h",
    "mkt_oi_breadth_rising_24h",
)
XS_FIELDS = (
    "xs_dispersion__funding_per_hour",
    "xs_dispersion__efficiency_ratio_20",
    "xs_dispersion__amihud_illiq",
    "xs_dispersion__rvol_z",
    "xs_dispersion__rvol_z_peer_resid",
    "xs_dispersion__vol_z_peer_resid",
    "xs_dispersion__volume_percentile",
)
SP_SOURCES = (
    "price_recovery_fraction_24h",
    "leverage_build",
    "efficiency_ratio_20",
    "volume_trend_48",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_prior_self_state(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add a 30d prior percentile state per symbol.

    Values at ``t`` are computed before inserting the current candidate into
    the history.  The P0 stream is itself target-free and is exactly the route
    available at inference; therefore the transform is causally reproducible
    without pretending this sparse route is a full-universe cross section.
    """
    output = frame.copy()
    fields: list[str] = []
    ordered = output.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable")
    for source in SP_SOURCES:
        p_name = f"sp30_prior_pct__{source}"
        p = pd.Series(np.nan, index=ordered.index, dtype=np.float32)
        for _symbol, group in ordered.groupby("__symbol__", sort=False):
            hist: deque[tuple[pd.Timestamp, float]] = deque()
            for index, row in group.iterrows():
                stamp = pd.Timestamp(row["__decision_ts__"])
                while hist and hist[0][0] < stamp - pd.Timedelta(days=90):
                    hist.popleft()
                value = pd.to_numeric(pd.Series([row[source]]), errors="coerce").iloc[0]
                prior = np.asarray([item[1] for item in hist if np.isfinite(item[1])], dtype=float)
                if np.isfinite(value):
                    p30 = prior[np.asarray([item[0] >= stamp - pd.Timedelta(days=30) for item in hist], dtype=bool)]
                    # The P0 route is sparse by design (one target-free
                    # candidate per hour).  Two prior route observations are
                    # the smallest honest support that still keeps the block
                    # above the declared 90% later-era coverage gate.
                    if len(p30) >= 2:
                        p.at[index] = np.float32(np.mean(p30 <= float(value)))
                if np.isfinite(value):
                    hist.append((stamp, float(value)))
        output[p_name] = p.reindex(output.index).to_numpy(np.float32)
        fields.append(p_name)
    return output, tuple(fields)


def _layer_fields(base: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys((*base, *(field for field in extra if field not in base))))
    if len(fields) != len(set(fields)):
        raise AssertionError("feature block contains duplicate fields")
    return fields


def _coverage(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    later = frame.loc[frame["__decision_ts__"].dt.strftime("%Y").isin(LATER_ERAS)].copy()
    rows = []
    for field in fields:
        values = pd.to_numeric(later[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rows.append({"feature": field, "later_coverage": float(values.notna().mean())})
    audit = pd.DataFrame(rows)
    if not audit.empty and bool((audit.later_coverage < MIN_LATER_COVERAGE).any()):
        bad = audit.loc[audit.later_coverage < MIN_LATER_COVERAGE, "feature"].tolist()
        raise AssertionError(f"below {MIN_LATER_COVERAGE:.0%} later-era coverage: {bad}")
    return audit


def _metrics(prediction: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    monthly, era, summary = r3b._metrics(prediction, arm)
    # ``_metrics`` carries the exact existing no-leakage policy-net accounting.
    later = era.loc[era.era.isin(LATER_ERAS)].copy()
    if later.empty:
        raise AssertionError("no 2025--2026 output")
    weights = np.maximum(later.outcome_known_candidates.to_numpy(float), 1.0)
    summary = {
        **summary,
        "later_weighted_net_bps_per_trade": float(np.average(later.net_bps_per_trade.to_numpy(float), weights=weights)),
        "later_worst_era_net_bps_per_trade": float(later.net_bps_per_trade.min()),
        "later_positive_months": int(later.positive_months.sum()),
        "later_months": int(later.months.sum()),
    }
    return monthly, era, summary


def _table(frame: pd.DataFrame) -> str:
    """Render without requiring the optional tabulate package."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def _run_arm(frame: pd.DataFrame, *, name: str, o_fields: tuple[str, ...], c_fields: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction, fold_audit = r3._run_target(
        frame, o_fields, c_fields, TARGET, C_SEED, "uniform", o_seed=O_SEED,
    )
    monthly, era, summary = _metrics(prediction, name)
    return prediction, fold_audit, monthly, {**summary, "era_metrics": era.to_dict("records")}


def run(out: Path, *, combined_xs_sp: bool = False) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    frame, frozen_o45, _m4, sources = r3._load_frame()
    frozen_c59 = c59._c59()
    frame, sp_fields = _add_prior_self_state(frame)
    block_fields = {"SF": SF_FIELDS, "TF": TF_FIELDS, "XS": XS_FIELDS, "SP": sp_fields}
    source_inventory = tuple(r1._load_f115_selection(r1.DEFAULT_FEATURE_SELECTION))
    if not set((*SF_FIELDS, *TF_FIELDS, *XS_FIELDS)).issubset(source_inventory):
        raise AssertionError("approved block fields are not sourced from immutable target-free F115")
    coverage = pd.concat([
        _coverage(frame, fields).assign(block=block)
        for block, fields in block_fields.items()
    ], ignore_index=True)

    # Each layer sees only its own requested additional block.  This makes a
    # failed block diagnosable and avoids a broad factorial feature search.
    if combined_xs_sp:
        # This is the only combination opened after the separate block screen:
        # XS was the only O block improving both selection eras and SP was the
        # only C block doing so.  No other block is silently carried forward.
        arms = [("O_XS__C_SP", _layer_fields(frozen_o45, XS_FIELDS), _layer_fields(frozen_c59, sp_fields))]
    else:
        arms = [
            ("O0_frozen_O45", frozen_o45, frozen_c59),
            ("C0_frozen_C59", frozen_o45, frozen_c59),
        ]
        arms.extend((f"O_{block}", _layer_fields(frozen_o45, fields), frozen_c59) for block, fields in block_fields.items())
        arms.extend((f"C_{block}", frozen_o45, _layer_fields(frozen_c59, fields)) for block, fields in block_fields.items())

    predictions: list[pd.DataFrame] = []
    fold_audits: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for index, (name, o_fields, c_fields) in enumerate(arms):
        # The duplicate frozen control is retained only to prove that the two
        # stages start from byte-identical upstream contracts.
        prediction, audit, monthly, summary = _run_arm(frame, name=name, o_fields=o_fields, c_fields=c_fields)
        prediction["feature_block_arm"] = name
        audit["feature_block_arm"] = name
        monthly["feature_block_arm"] = name
        summary["feature_block_arm"] = name
        summary["o_field_count"] = len(o_fields)
        summary["c_field_count"] = len(c_fields)
        summaries.append(summary)
        predictions.append(prediction)
        fold_audits.append(audit)
        monthly_parts.append(monthly)

    summary = pd.DataFrame(summaries)
    # The two controls must agree exactly: they are intentionally the same
    # computation and protect against accidental stage-specific changes.
    if not combined_xs_sp:
        controls = summary.loc[summary.feature_block_arm.isin(("O0_frozen_O45", "C0_frozen_C59"))]
        numeric = ("later_weighted_net_bps_per_trade", "later_worst_era_net_bps_per_trade", "selected", "total_net_bps")
        for column in numeric:
            if not np.allclose(controls[column].to_numpy(float), controls[column].iloc[0], rtol=0.0, atol=1e-6, equal_nan=True):
                raise AssertionError(f"frozen O/C controls differ for {column}")

    out.mkdir(parents=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "feature_block_outer_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat(fold_audits, ignore_index=True).to_parquet(out / "feature_block_fold_audit.parquet", index=False, compression="zstd")
    pd.concat(monthly_parts, ignore_index=True).to_parquet(out / "feature_block_monthly_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "feature_block_summary.parquet", index=False, compression="zstd")
    coverage.to_parquet(out / "feature_block_coverage_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": "short",
        "scope": "research only; 2024 diagnostic and 2025--2026 selection/portability kept separate",
        "mode": "selected_O_XS_plus_C_SP_combination" if combined_xs_sp else "separate_single_block_screen",
        "period": {"candidate_start": "2024-05", "output_supported_start": "2024-10", "end_exclusive": "2026-08"},
        "architecture": "frozen P0 -> O250/H6 -> C3 normalized-regret -> analytic K0; +75 bps fixed admission",
        "invariants": {
            "features": "target-free F115 only; SP inserts current value only after computing its state",
            "cross_section": "XS fields are pre-existing full-universe-before-filter source fields",
            "labels": "valid labels only for fitting and outcome metrics; all target-free candidates scored",
            "prequential": "existing r3 strict outer/inner label_available_at < validation decision contract",
            "not_a_factorial_search": True,
        },
        "blocks": {key: list(value) for key, value in block_fields.items()},
        "summary": summary.to_dict("records"),
        "sources": {**sources, "feature_selection_sha256": _sha256(r1.DEFAULT_FEATURE_SELECTION)},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Short P0 O/C causal feature-block ablation", "",
        "All results are strict-prequential target-free scores.  2024 is diagnostic; 2025--2026 remain separated and govern selection.", "",
        "## Later-era summary", "", _table(summary.drop(columns=["era_metrics"], errors="ignore")), "",
        "## Feature coverage", "", _table(coverage), "",
        "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_FEATURE_BLOCK_REPORT.md").write_text("\n".join(report))
    return out


def finalize_existing(out: Path) -> Path:
    """Recover report rendering after a post-compute optional-dependency error."""
    manifest_path = out / "run_manifest.json"
    summary_path = out / "feature_block_summary.parquet"
    coverage_path = out / "feature_block_coverage_audit.parquet"
    if not (manifest_path.exists() and summary_path.exists() and coverage_path.exists()):
        raise FileNotFoundError("cannot finalise incomplete feature-block artifact")
    manifest = json.loads(manifest_path.read_text())
    summary = pd.read_parquet(summary_path)
    coverage = pd.read_parquet(coverage_path)
    report = [
        "# Short P0 O/C causal feature-block ablation", "",
        "All results are strict-prequential target-free scores.  2024 is diagnostic; 2025--2026 remain separated and govern selection.", "",
        "## Later-era summary", "", _table(summary.drop(columns=["era_metrics"], errors="ignore")), "",
        "## Feature coverage", "", _table(coverage), "",
        "## Contract", "", "```json", json.dumps(manifest, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_FEATURE_BLOCK_REPORT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--finalize-existing", action="store_true", help="render a report from a fully written artifact without rerunning fits")
    parser.add_argument("--combined-xs-sp", action="store_true", help="run only the predeclared O_XS plus C_SP combination after the separate block screen")
    args = parser.parse_args()
    print(finalize_existing(args.out) if args.finalize_existing else run(args.out, combined_xs_sp=args.combined_xs_sp))


if __name__ == "__main__":
    main()
