#!/usr/bin/env python3
"""Audit whether native L2 continuation features can join candidate panels.

This is deliberately a readiness audit, not a model runner.  It reads only
candidate identity/timestamp fields and native L2 availability metadata.  It
does not load labels, scores, or outcome-derived columns, and it never fills a
feature forward beyond the declared staleness window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_L2 = ROOT / "data_perp/artifacts/native_l2_continuation_sidecar_20260801_v2"
DEFAULT_OUT = ROOT / "data_perp/artifacts/native_l2_candidate_overlap_audit_20260801_v2"

PANEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "panel_id": "canonical_candidate_handoff",
        "path": "data_perp/artifacts/execution_ev_canonical_alpha_inputs_july20_20260726_v1/candidate_handoff.parquet",
        "time_column": "available_at",
        "time_semantics": "candidate_feature_availability_time",
    },
    {
        "panel_id": "july20_23_retrospective_candidate_bridge",
        "path": "data_perp/artifacts/july20_23_retrospective_allscore_bridge_20260730_v1/retrospective_allscore_bridge.parquet",
        "time_column": "__ts__",
        "time_semantics": "candidate_observation_time",
    },
    {
        "panel_id": "exact_h12_side_local_residual_oof",
        "path": "data_perp/artifacts/exact_h12_side_local_residual_oof_20260730_v1/oof_predictions.parquet",
        "time_column": "__ts__",
        "time_semantics": "candidate_observation_time",
    },
    {
        "panel_id": "a_grade_strict_forward_candidate_scores",
        "path": "data_perp/artifacts/a_grade_cost_clearing_conversion_ablation_20260730_v6/strict_forward_2026_candidate_scores.parquet",
        "time_column": "__ts__",
        "time_semantics": "candidate_observation_time",
    },
)

STALENESS_SECONDS = (0.0, 3600.0, 7200.0, 21600.0, 86400.0)


def _normalise_symbol(value: Any) -> str:
    """Map native file keys and candidate keys to exact product identity.

    ``*_USD_USD`` is the Kraken file-key encoding for ``*/USD:USD``.  The
    collateral suffix is part of identity; it is therefore not collapsed into
    a base-asset-only key.
    """

    text = str(value or "").upper().strip()
    if not text:
        return ""
    for suffix, replacement in (
        ("_USD_USD", "/USD:USD"),
        ("_USD_BTC", "/USD:BTC"),
        ("_USD_ETH", "/USD:ETH"),
        ("_USD_LTC", "/USD:LTC"),
        ("_USD_XRP", "/USD:XRP"),
        ("_USD:USD", "/USD:USD"),
        ("_USD:BTC", "/USD:BTC"),
        ("_USD:ETH", "/USD:ETH"),
        ("_USD:LTC", "/USD:LTC"),
        ("_USD:XRP", "/USD:XRP"),
    ):
        if text.endswith(suffix):
            return text[: -len(suffix)] + replacement
    return text.replace("_", "/")


def _read_l2(l2_path: Path) -> pd.DataFrame:
    path = l2_path / "native_l2_continuation_features.parquet"
    required = [
        "symbol",
        "snapshot_ts",
        "feature_available_at",
        "l2_snapshot_gap_seconds",
        "l2_mid_return_prev_snapshot",
    ]
    table = pq.read_table(path, columns=required)
    frame = table.to_pandas()
    frame["symbol_norm"] = frame["symbol"].map(_normalise_symbol)
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
    frame["feature_available_at"] = pd.to_datetime(
        frame["feature_available_at"], utc=True, errors="coerce"
    )
    if frame[["symbol_norm", "snapshot_ts"]].isna().any().any():
        raise ValueError("native L2 sidecar has missing symbol or timestamp")
    if frame.duplicated(["symbol_norm", "snapshot_ts"]).any():
        raise ValueError("native L2 sidecar has duplicate symbol/timestamp keys")
    if (frame["feature_available_at"] > frame["snapshot_ts"]).any():
        raise ValueError("native L2 feature availability is after its snapshot")
    # ``l2_snapshot_gap_seconds`` is retained as a diagnostic even for large
    # gaps.  A row is lag-ready only when the bounded previous-snapshot
    # transform itself is present.
    frame["lag_features_ready"] = frame["l2_mid_return_prev_snapshot"].notna()
    return frame.sort_values(["snapshot_ts", "symbol_norm"], kind="stable").reset_index(drop=True)


def _read_panel(spec: dict[str, str]) -> pd.DataFrame:
    path = ROOT / spec["path"]
    if not path.exists():
        return pd.DataFrame()
    names = set(pq.ParquetFile(path).schema.names)
    if "candidate_id" not in names:
        raise ValueError(f"{path} has no candidate_id")
    if "__symbol__" in names:
        symbol_column = "__symbol__"
    elif "symbol" in names:
        symbol_column = "symbol"
    else:
        raise ValueError(f"{path} has no symbol column")
    time_column = spec["time_column"]
    if time_column not in names:
        raise ValueError(f"{path} has no declared timestamp column {time_column!r}")
    # Keep identity/timing only.  In particular, labels, scores, and costs are
    # intentionally not read into this audit.
    cols = ["candidate_id", symbol_column, time_column]
    if "side_name" in names:
        cols.append("side_name")
    table = pq.read_table(path, columns=list(dict.fromkeys(cols)))
    frame = table.to_pandas()
    frame = frame.rename(columns={symbol_column: "source_symbol", time_column: "candidate_ts"})
    frame["source_symbol"] = frame["source_symbol"].map(_normalise_symbol)
    frame["candidate_ts"] = pd.to_datetime(frame["candidate_ts"], utc=True, errors="coerce")
    frame = frame.loc[frame["candidate_id"].notna() & frame["source_symbol"].ne("") & frame["candidate_ts"].notna()].copy()
    # Some research panels intentionally repeat a candidate across ablation
    # arms.  Preserve those rows; overlap is assessed per panel row and does
    # not claim that the panel itself is a unique candidate ledger.
    return frame.sort_values(["candidate_ts", "source_symbol"], kind="stable").reset_index(drop=True)


def _asof_join(panel: pd.DataFrame, l2: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    joined = pd.merge_asof(
        panel,
        l2[["symbol_norm", "snapshot_ts", "feature_available_at", "lag_features_ready"]],
        left_on="candidate_ts",
        right_on="snapshot_ts",
        left_by="source_symbol",
        right_by="symbol_norm",
        direction="backward",
        allow_exact_matches=True,
    )
    if (joined["snapshot_ts"].notna() & (joined["snapshot_ts"] > joined["candidate_ts"])).any():
        raise ValueError("as-of join produced a future native L2 snapshot")
    joined["matched_snapshot_age_seconds"] = (
        joined["candidate_ts"] - joined["snapshot_ts"]
    ).dt.total_seconds()
    joined["native_snapshot_match"] = joined["snapshot_ts"].notna()
    return joined


def _coverage_rows(panel_id: str, joined: pd.DataFrame) -> list[dict[str, Any]]:
    if joined.empty:
        return [{"panel_id": panel_id, "staleness_hours": h / 3600.0, "rows": 0, "matched_rows": 0, "coverage": 0.0, "symbols": 0, "days": 0, "lag_ready_rows": 0, "lag_ready_coverage": 0.0} for h in STALENESS_SECONDS]
    rows: list[dict[str, Any]] = []
    for max_age in STALENESS_SECONDS:
        valid = joined["native_snapshot_match"] & joined["matched_snapshot_age_seconds"].le(max_age)
        lag_ready = valid & joined["lag_features_ready"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        rows.append(
            {
                "panel_id": panel_id,
                "staleness_hours": max_age / 3600.0,
                "rows": int(len(joined)),
                "matched_rows": int(valid.sum()),
                "coverage": float(valid.mean()),
                "symbols": int(joined.loc[valid, "source_symbol"].nunique()),
                "days": int(joined.loc[valid, "candidate_ts"].dt.floor("D").nunique()),
                "lag_ready_rows": int(lag_ready.sum()),
                "lag_ready_coverage": float(lag_ready.mean()),
            }
        )
    return rows


def _write_report(
    out: Path,
    l2: pd.DataFrame,
    coverage: pd.DataFrame,
    available_panels: list[dict[str, Any]],
    max_staleness_hours: float,
) -> None:
    accepted = coverage.loc[coverage["staleness_hours"].eq(max_staleness_hours)]
    lines = [
        "# Native L2 candidate-overlap audit",
        "",
        "Status: `RESEARCH_ONLY_OVERLAP_READINESS_NO_MODEL`",
        "",
        "This audit uses only candidate identity/timestamp fields and the native Kraken L2 availability sidecar. It does not load labels, scores, costs, or portfolio constraints. The join is backward/as-of by exact product identity; no future snapshot or forward fill is permitted.",
        "",
        "## Native source",
        "",
        f"- Native snapshots: **{len(l2):,}** rows across **{l2['symbol_norm'].nunique():,}** exact product identities.",
        f"- Coverage: **{l2['snapshot_ts'].min().isoformat()}** to **{l2['snapshot_ts'].max().isoformat()}**.",
        f"- Accepted diagnostic staleness: **{max_staleness_hours:g} hours**; this is not a production promotion decision.",
        "- All `local_ohlcv_summary` proxy rows were excluded from the sidecar.",
        "",
        "## Candidate overlap at the declared as-of join",
        "",
        "| panel | rows | matched rows | coverage | symbols | days | rows with prior-snapshot lag features | lag-ready coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in accepted.sort_values("panel_id").iterrows():
        lines.append(
            f"| {row.panel_id} | {int(row.rows):,} | {int(row.matched_rows):,} | {row.coverage:.3%} | {int(row.symbols):,} | {int(row.days):,} | {int(row.lag_ready_rows):,} | {row.lag_ready_coverage:.3%} |"
        )
    nonzero = accepted.loc[accepted["matched_rows"].gt(0), "panel_id"].tolist()
    zero = accepted.loc[accepted["matched_rows"].eq(0), "panel_id"].tolist()
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A matched row only means a native snapshot existed at or before the candidate timestamp and within the declared staleness bound. It does not establish sufficient economic support for training.",
            f"- Non-zero overlap at the declared bound: {', '.join(nonzero) if nonzero else 'none'}.",
            f"- Zero overlap at the declared bound: {', '.join(zero) if zero else 'none'}.",
            "- The current cohort still cannot support a full historical strict-OOF experiment because the exact-H12 May–July panels have no overlap and the canonical handoff is only partially covered.",
            "- Therefore no strict OOF model, feature-selection result, or policy/economic claim is promoted from native L2 yet. The next admissible step is to obtain a longer exact native-L2 history (or a precisely timestamped native feed) and rerun the same as-of join on the full candidate population.",
            "",
            "## Fail-closed gates",
            "",
            "- `candidate_joined`: false for production; this artifact is a readiness audit only.",
            "- `labels_used`: false.",
            "- `promotion_eligible`: false.",
            "- `portfolio_constraints_in_scope`: false.",
        ]
    )
    (out / "NATIVE_L2_CANDIDATE_OVERLAP_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2-dir", type=Path, default=DEFAULT_L2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-staleness-hours", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_staleness_hours < 0:
        raise SystemExit("--max-staleness-hours must be non-negative")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    l2 = _read_l2(args.l2_dir)
    coverage_rows: list[dict[str, Any]] = []
    day_rows: list[pd.DataFrame] = []
    symbol_rows: list[pd.DataFrame] = []
    available: list[dict[str, Any]] = []
    primary_rows: pd.DataFrame | None = None
    for spec in PANEL_SPECS:
        panel = _read_panel(spec)
        if panel.empty:
            continue
        joined = _asof_join(panel, l2)
        coverage_rows.extend(_coverage_rows(spec["panel_id"], joined))
        available.append(
            {
                "panel_id": spec["panel_id"],
                "path": spec["path"],
                "time_column": spec["time_column"],
                "time_semantics": spec["time_semantics"],
                "rows": int(len(panel)),
                "symbols": int(panel["source_symbol"].nunique()),
                "min_candidate_ts": panel["candidate_ts"].min().isoformat(),
                "max_candidate_ts": panel["candidate_ts"].max().isoformat(),
            }
        )
        bound = joined["native_snapshot_match"] & joined["matched_snapshot_age_seconds"].le(args.max_staleness_hours * 3600.0)
        temp = joined.loc[bound, ["source_symbol", "candidate_ts", "matched_snapshot_age_seconds", "lag_features_ready"]].copy()
        if not temp.empty:
            temp["panel_id"] = spec["panel_id"]
            temp["day"] = temp["candidate_ts"].dt.floor("D")
            day_rows.append(
                temp.groupby(["panel_id", "day"], as_index=False).agg(
                    matched_rows=("candidate_ts", "size"),
                    symbols=("source_symbol", "nunique"),
                    lag_ready_rows=("lag_features_ready", "sum"),
                    median_age_seconds=("matched_snapshot_age_seconds", "median"),
                )
            )
            symbol_rows.append(
                temp.groupby(["panel_id", "source_symbol"], as_index=False).agg(
                    matched_rows=("candidate_ts", "size"),
                    days=("day", "nunique"),
                    lag_ready_rows=("lag_features_ready", "sum"),
                    median_age_seconds=("matched_snapshot_age_seconds", "median"),
                )
            )
        if spec["panel_id"] == "canonical_candidate_handoff":
            primary_rows = joined.assign(
                accepted_asof_2h=bound,
            )[
                [
                    "candidate_id",
                    "source_symbol",
                    "candidate_ts",
                    "snapshot_ts",
                    "matched_snapshot_age_seconds",
                    "native_snapshot_match",
                    "lag_features_ready",
                    "accepted_asof_2h",
                ]
            ]

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / "candidate_overlap_coverage.csv", index=False)
    pd.DataFrame(available).to_csv(out / "candidate_panel_inventory.csv", index=False)
    if day_rows:
        pd.concat(day_rows, ignore_index=True).to_csv(out / "candidate_overlap_by_day.csv", index=False)
    else:
        pd.DataFrame(columns=["panel_id", "day", "matched_rows", "symbols", "lag_ready_rows", "median_age_seconds"]).to_csv(out / "candidate_overlap_by_day.csv", index=False)
    if symbol_rows:
        pd.concat(symbol_rows, ignore_index=True).to_csv(out / "candidate_overlap_by_symbol.csv", index=False)
    else:
        pd.DataFrame(columns=["panel_id", "source_symbol", "matched_rows", "days", "lag_ready_rows", "median_age_seconds"]).to_csv(out / "candidate_overlap_by_symbol.csv", index=False)
    if primary_rows is not None:
        primary_rows.to_parquet(out / "canonical_handoff_overlap_rows.parquet", index=False)

    _write_report(
        out,
        l2,
        coverage,
        available,
        args.max_staleness_hours,
    )
    manifest = {
        "status": "RESEARCH_ONLY_OVERLAP_READINESS_NO_MODEL",
        "promotion_eligible": False,
        "candidate_joined": False,
        "labels_used": False,
        "portfolio_constraints_in_scope": False,
        "native_l2_sidecar": str((args.l2_dir / "native_l2_continuation_features.parquet").resolve()),
        "join": "backward_asof_exact_product_identity_no_forward_fill",
        "max_staleness_hours": float(args.max_staleness_hours),
        "native_l2_rows": int(len(l2)),
        "native_l2_symbols": int(l2["symbol_norm"].nunique()),
        "panels": available,
        "coverage_file": "candidate_overlap_coverage.csv",
        "report": "NATIVE_L2_CANDIDATE_OVERLAP_AUDIT.md",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
