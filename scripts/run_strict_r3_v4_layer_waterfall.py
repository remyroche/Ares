#!/usr/bin/env python3
"""Materialise a strict-R3 v4 long-only layer waterfall.

Two evaluation views are deliberately kept separate:

* global top-k tails are *retrospective rank diagnostics* only; and
* the causal 21-day expected-net map is the executable admission layer.

Every row in the output is a held prediction from a chronological OOF block.
The policy outcome is the frozen SimplePolicyOptimiser label already present
in the strict-R3 ledger: next-bar entry, H12 timeout, 100-bps cost once.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_n5_canonical_selection import _load_selection_input  # noqa: E402


SCHEMA = "strict_r3_v4_layer_waterfall_v2"
TAILS = (0.005, 0.01, 0.02, 0.03, 0.05)
YEARS = (2025, 2026)
SOURCE = {
    2025: ROOT / "data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2025_janjul_20260810_v1/predictions.parquet",
    2026: ROOT / "data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2026_janjul_20260810_v1/predictions.parquet",
}
SIDECAR = {
    2025: ROOT / "data_perp/artifacts/strict_r3_ldf_canonical45_legacy_score_features_2025_20260811_v1/n5_causal_features.parquet",
    2026: ROOT / "data_perp/artifacts/strict_r3_ldf_canonical45_legacy_score_features_2026_20260811_v1/n5_causal_features.parquet",
}
LDF_OOF = {
    2025: ROOT / "data_perp/artifacts/strict_r3_ldf_mda_legacy_score_compact12_hpo_20260811_v1/oof_predictions_2025.parquet",
    2026: ROOT / "data_perp/artifacts/strict_r3_ldf_mda_legacy_score_compact12_hpo_20260811_v1/oof_predictions_2026.parquet",
}
FEATURE_CONTRACT = ROOT / "config/strict_r3_ldf_support_v4.json"


def _load_selected_conversion_contract(year: int) -> dict[str, Any]:
    """Return the active conversion overlay, rejecting stale attribution.

    The current C3 winner is intentionally the policy-residual correctness
    overlay.  Severe-200 was evaluated on its frozen TP6/SL4 definition but
    failed the 2026 transport gate, so it is only a shadow diagnostic.  This
    guard prevents a later waterfall from labelling an identity transform as a
    live Severe demotion merely because a legacy ``raw_severe`` column is
    present in the source ledger.
    """

    manifest_path = SOURCE[year].parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    overlays = manifest.get("overlay_arms")
    if not isinstance(overlays, list) or len(overlays) != 1:
        raise ValueError(f"{year} source must declare exactly one selected overlay")
    overlay = overlays[0]
    expected = {
        "name": "correctness_top30_k9temp025_no_memberships",
        "severe_target": "none",
        "severe_alpha": 0.0,
        "use_correctness": True,
        "correctness_training_fraction": 0.30,
        "k9_soft_memberships": False,
    }
    for key, value in expected.items():
        observed = overlay.get(key)
        if isinstance(value, float):
            if not np.isclose(float(observed), value, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"{year} source overlay {key!r}={observed!r}, expected {value!r}",
                )
        elif observed != value:
            raise ValueError(
                f"{year} source overlay {key!r}={observed!r}, expected {value!r}",
            )
    if manifest.get("score_override_arm") != "conditional_none":
        raise ValueError(f"{year} source is not the selected conditional-none score")
    return {
        "source_manifest": str(manifest_path),
        "selected_overlay": dict(overlay),
        "severe_active": False,
        "score_modulator": "top30_policy_residual_correctness",
    }


def _valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
    )


def _top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(float(fraction) * len(frame))))
    return frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(count)


def _metric_row(
    selected: pd.DataFrame,
    *,
    population_rows: int,
    stage: str,
    selection: str,
    tail: float | None,
    size_column: str | None = None,
) -> dict[str, Any]:
    valid = selected.loc[_valid(selected)].copy()
    size = (
        pd.to_numeric(valid[size_column], errors="coerce").fillna(1.0).to_numpy(float)
        if size_column is not None
        else np.ones(len(valid), dtype=float)
    )
    net = pd.to_numeric(valid["policy_net_bps"], errors="coerce").to_numpy(float)
    gross = pd.to_numeric(valid["policy_gross_bps"], errors="coerce").to_numpy(float)
    weighted_net = float(np.average(net, weights=size)) if len(valid) and size.sum() > 0.0 else np.nan
    weighted_gross = float(np.average(gross, weights=size)) if len(valid) and size.sum() > 0.0 else np.nan
    return {
        "stage": stage,
        "selection": selection,
        "tail": tail,
        "population_rows": int(population_rows),
        "selected_score_rows": int(len(selected)),
        "valid_outcomes": int(len(valid)),
        "outcome_coverage": float(len(valid) / max(len(selected), 1)),
        "net_bps_per_trade": float(np.mean(net)) if len(valid) else np.nan,
        "gross_bps_per_trade": float(np.mean(gross)) if len(valid) else np.nan,
        "exposure_weighted_net_bps": weighted_net,
        "exposure_weighted_gross_bps": weighted_gross,
        "positive_rate": float(np.mean(net > 0.0)) if len(valid) else np.nan,
        "mean_size_multiplier": float(np.mean(size)) if len(valid) else 1.0,
    }


def _diagnostic_rows(frame: pd.DataFrame, *, stage: str, score: str, size_column: str | None) -> list[dict[str, Any]]:
    return [
        _metric_row(
            _top(frame, score, tail),
            population_rows=len(frame),
            stage=stage,
            selection="retrospective_global_tail_diagnostic",
            tail=tail,
            size_column=size_column,
        )
        for tail in TAILS
    ]


def _time_breakdown(selected: pd.DataFrame, metric: dict[str, Any], frequency: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    work = selected.copy()
    work["period"] = pd.to_datetime(work["__decision_ts__"], utc=True).dt.to_period(frequency).astype(str)
    rows: list[dict[str, Any]] = []
    for period, block in work.groupby("period", sort=True):
        row = _metric_row(
            block,
            population_rows=int(metric["population_rows"]),
            stage=str(metric["stage"]),
            selection=str(metric["selection"]),
            tail=metric["tail"],
            size_column="trust_size_multiplier" if str(metric["stage"]) == "ldf_sizing" else None,
        )
        row["period"] = period
        rows.append(row)
    return pd.DataFrame(rows)


def _require_policy_cost_once(frame: pd.DataFrame) -> None:
    valid = frame.loc[_valid(frame)]
    if valid.empty:
        raise ValueError("waterfall has no valid policy outcomes")
    cost = (
        pd.to_numeric(valid["policy_gross_bps"], errors="coerce")
        - pd.to_numeric(valid["policy_net_bps"], errors="coerce")
    )
    if not np.allclose(cost.to_numpy(float), 100.0, atol=1e-5, rtol=0.0):
        raise AssertionError("frozen policy outcome does not apply exactly 100 bps cost once")


def _load_year(year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    conversion_contract = _load_selected_conversion_contract(year)
    frame, _fields, _audit = _load_selection_input(
        SOURCE[year],
        feature_sidecar=SIDECAR[year],
        feature_contract=FEATURE_CONTRACT,
    )
    ldf = pd.read_parquet(LDF_OOF[year])
    if ldf["candidate_id"].duplicated().any():
        raise ValueError(f"{year} LDF OOF has duplicate candidate IDs")
    columns = ["candidate_id", "trust_size_multiplier"]
    frame = frame.merge(ldf.loc[:, columns], on="candidate_id", how="inner", validate="one_to_one")
    # The LDF loader intentionally keeps only its own feature contract.  The
    # waterfall needs immutable upstream layer scores as *audit fields*, so
    # reattach them from the same source ledger by identity.  They are never
    # fed into the admission map or LDF here.
    upstream_columns = [
        "candidate_id",
        "stack_is_prequential",
        "prequential_base_rank42",
        "prequential_consensus_rank",
        "prequential_upstream",
        "raw_severe",
    ]
    upstream = pd.read_parquet(SOURCE[year], columns=upstream_columns)
    if upstream["candidate_id"].duplicated().any():
        raise ValueError(f"{year} upstream ledger has duplicate candidate IDs")
    frame = frame.merge(upstream, on="candidate_id", how="left", validate="one_to_one")
    if frame.empty:
        raise ValueError(f"{year} waterfall has no LDF OOF intersection")
    if not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise AssertionError(f"{year} waterfall contains non-prequential upstream rows")
    # This selected source deliberately keeps the legacy field for parity with
    # the ablation ledger, but it must be an exact identity if the source says
    # Severe is shadow-only.  Do not present it as a layer contribution.
    if not np.allclose(
        pd.to_numeric(frame["raw_severe"], errors="coerce").to_numpy(float),
        pd.to_numeric(frame["prequential_upstream"], errors="coerce").to_numpy(float),
        equal_nan=True,
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError(
            f"{year} source claims shadow-only Severe but raw_severe changes the score",
        )
    _require_policy_cost_once(frame)
    return (
        frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True),
        conversion_contract,
    )


def _admission_rows(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"causal admission fields missing: {missing}")
    admitted = frame.loc[
        frame["causal_21d_side_admitted_ge_50bps"].fillna(False).astype(bool)
        & frame["causal_21d_side_expected_net_bps"].notna()
    ].copy()
    row = _metric_row(
        admitted,
        population_rows=len(frame),
        stage="causal_ev_admission",
        selection="executable_21d_side_local_ev_ge_50bps",
        tail=None,
        size_column="trust_size_multiplier",
    )
    row.update(
        admission_rate=float(len(admitted) / len(frame)),
        mapped_expected_net_bps=float(
            pd.to_numeric(admitted["causal_21d_side_expected_net_bps"], errors="coerce").mean(),
        ) if len(admitted) else np.nan,
        trades_per_day=float(len(admitted) / max(frame["__decision_ts__"].dt.normalize().nunique(), 1)),
    )
    return row, admitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    global_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    admission_rows: list[dict[str, Any]] = []
    admission_monthly: list[pd.DataFrame] = []
    score_stages = (
        ("base_rank42", "prequential_base_rank42", None),
        ("consensus_rank", "prequential_consensus_rank", None),
        ("base_consensus_75_25", "prequential_upstream", None),
        ("correctness_top30_prior42_cdf", "final_score", None),
        ("ldf_sizing", "final_score", "trust_size_multiplier"),
    )
    conversion_contracts: dict[str, Any] = {}
    for year in YEARS:
        frame, conversion_contract = _load_year(year)
        conversion_contracts[str(year)] = conversion_contract
        for stage, score, size in score_stages:
            for metric in _diagnostic_rows(frame, stage=stage, score=score, size_column=size):
                metric["year"] = year
                global_rows.append(metric)
                selected = _top(frame, score, float(metric["tail"]))
                monthly_rows.append(_time_breakdown(selected, metric, "M").assign(year=year))
                weekly_rows.append(_time_breakdown(selected, metric, "W-MON").assign(year=year))
        admission, admitted = _admission_rows(frame)
        admission["year"] = year
        admission_rows.append(admission)
        admitted["period"] = admitted["__decision_ts__"].dt.to_period("M").astype(str)
        admission_monthly.append(
            admitted.groupby("period", sort=True).apply(
                lambda block: pd.Series(
                    {
                        "admitted_rows": int(len(block)),
                        "valid_outcomes": int(_valid(block).sum()),
                        "net_bps_per_trade": float(pd.to_numeric(block.loc[_valid(block), "policy_net_bps"], errors="coerce").mean()),
                        "mapped_expected_net_bps": float(pd.to_numeric(block["causal_21d_side_expected_net_bps"], errors="coerce").mean()),
                    },
                ),
                include_groups=False,
            ).reset_index().assign(year=year),
        )
        frame.to_parquet(args.out_dir / f"waterfall_ledger_{year}.parquet", index=False, compression="zstd")

    pd.DataFrame(global_rows).to_parquet(args.out_dir / "waterfall_global.parquet", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(args.out_dir / "waterfall_monthly.parquet", index=False)
    pd.concat(weekly_rows, ignore_index=True).to_parquet(args.out_dir / "waterfall_weekly.parquet", index=False)
    pd.DataFrame(admission_rows).to_parquet(args.out_dir / "admission_global.parquet", index=False)
    pd.concat(admission_monthly, ignore_index=True).to_parquet(args.out_dir / "admission_monthly.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "side": "long",
                "years": list(YEARS),
                "score_stages": [item[0] for item in score_stages],
                "global_tail_interpretation": "retrospective diagnostic only; not a live admission rule",
                "live_admission": "causal 21-day side-local expected policy net >= 50 bps; fail closed without map support",
                "policy": "frozen SimplePolicyOptimiser outcome: next-bar entry, H12 timeout, 100 bps cost once",
                "ldf": "current compact12 two-forest sizing OOF multiplier; no ranking or admission change",
                "conversion_overlay": (
                    "policy-residual correctness top-30% curriculum; "
                    "Severe-200 is shadow-only and excluded from score attribution"
                ),
                "conversion_contracts": conversion_contracts,
                "source": {str(year): str(SOURCE[year]) for year in YEARS},
                "ldf_oof": {str(year): str(LDF_OOF[year]) for year in YEARS},
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
