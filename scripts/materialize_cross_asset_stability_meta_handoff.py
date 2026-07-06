#!/usr/bin/env python3
"""Materialize leakage-safe cross-asset stability priors into meta handoff.

The promoted cross-asset representation is useful but unstable by month/cell.
This materializer adds prior-history stability context as model features, not
hard gates.  For each row, the added features are computed only from OOF
baseline-vs-promoted diagnostics in strictly earlier months for the same
side x source_semantic_family cell.

Rows with no prior history keep NaN priors, preserving the no-backfill contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_month_flip_attribution import (  # noqa: E402
    DEFAULT_BASELINE_SMOKE_DIR,
    DEFAULT_PROMOTED_HANDOFF_DIR,
    DEFAULT_PROMOTED_SMOKE_DIR,
    DEFAULT_OUT_DIR as DEFAULT_FLIP_AUDIT_DIR,
    run_audit as run_flip_audit,
)
from scripts.materialize_promoted_cross_asset_meta_handoff import (  # noqa: E402
    CONTRACT_NAME,
    HANDOFF_NAME,
    LEDGER_NAME,
    _json_safe,
)


DEFAULT_SOURCE_HANDOFF_DIR = DEFAULT_PROMOTED_HANDOFF_DIR
DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR.parent / "train_meta_handoff_promoted_cross_asset_stability_v1"
CELL_COLUMNS = ("side_name", "source_semantic_family")
STABILITY_KEEP_FRACS = (0.10, 0.20, 0.30)
PRIOR_METRICS = (
    "effect_value_score",
    "delta_ev_after_1pct",
    "delta_exec_margin",
    "delta_clean_exec_precision",
    "delta_full_path_bad_mae",
    "delta_timeout",
    "delta_mfe_before_mae",
    "delta_mae_before_mfe",
    "delta_cell_oracle_overlap",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_or_build_month_cells(
    *,
    flip_audit_dir: Path,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
) -> pd.DataFrame:
    cells_path = flip_audit_dir / "promoted_cross_asset_month_cell_effects.csv"
    if not cells_path.exists():
        run_flip_audit(
            baseline_smoke_dir=baseline_smoke_dir,
            promoted_smoke_dir=promoted_smoke_dir,
            out_dir=flip_audit_dir,
        )
    if not cells_path.exists():
        raise FileNotFoundError(cells_path)
    cells = pd.read_csv(cells_path)
    required = {"month", "keep_frac", *CELL_COLUMNS, *PRIOR_METRICS}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"Month-cell effects missing required columns: {missing}")
    return cells


def _prefix(keep_frac: float) -> str:
    return f"xastab_k{int(round(float(keep_frac) * 100)):03d}"


def _prior_feature_rows(month_cells: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cells = month_cells.copy()
    cells["month"] = cells["month"].astype(str)
    for month in months:
        history = cells[cells["month"].astype(str).lt(str(month))]
        current_keys = cells[cells["month"].astype(str).eq(str(month))][list(CELL_COLUMNS)].drop_duplicates()
        if current_keys.empty:
            continue
        for _, key_row in current_keys.iterrows():
            key = {col: str(key_row[col]) for col in CELL_COLUMNS}
            record: dict[str, Any] = {"month": str(month), **key}
            key_mask = pd.Series(True, index=history.index)
            for col, value in key.items():
                key_mask &= history[col].astype(str).eq(value)
            hist_cell = history.loc[key_mask]
            for keep in STABILITY_KEEP_FRACS:
                hist_keep = hist_cell[hist_cell["keep_frac"].astype(float).round(6).eq(float(keep))]
                pfx = _prefix(keep)
                record[f"{pfx}_history_months"] = int(hist_keep["month"].nunique()) if not hist_keep.empty else 0
                record[f"{pfx}_has_prior"] = float(1.0 if not hist_keep.empty else 0.0)
                if hist_keep.empty:
                    for metric in PRIOR_METRICS:
                        record[f"{pfx}_{metric}_prior_mean"] = np.nan
                        record[f"{pfx}_{metric}_prior_last"] = np.nan
                    record[f"{pfx}_beneficial_prior_rate"] = np.nan
                    record[f"{pfx}_damaged_prior_rate"] = np.nan
                    record[f"{pfx}_prior_value_volatility"] = np.nan
                    record[f"{pfx}_prior_value_positive_rate"] = np.nan
                    continue
                hist_keep = hist_keep.sort_values("month")
                for metric in PRIOR_METRICS:
                    values = pd.to_numeric(hist_keep[metric], errors="coerce")
                    record[f"{pfx}_{metric}_prior_mean"] = float(values.mean()) if values.notna().any() else np.nan
                    record[f"{pfx}_{metric}_prior_last"] = float(values.iloc[-1]) if len(values) and pd.notna(values.iloc[-1]) else np.nan
                beneficial = hist_keep.get("promoted_beneficial")
                damaged = hist_keep.get("promoted_damaged")
                value = pd.to_numeric(hist_keep.get("effect_value_score"), errors="coerce")
                record[f"{pfx}_beneficial_prior_rate"] = float(pd.to_numeric(beneficial, errors="coerce").mean()) if beneficial is not None else np.nan
                record[f"{pfx}_damaged_prior_rate"] = float(pd.to_numeric(damaged, errors="coerce").mean()) if damaged is not None else np.nan
                record[f"{pfx}_prior_value_volatility"] = float(value.std(ddof=0)) if value.notna().sum() >= 2 else 0.0
                record[f"{pfx}_prior_value_positive_rate"] = float(value.gt(0.0).mean()) if value.notna().any() else np.nan
            rows.append(record)
    return pd.DataFrame(rows)


def materialize(
    *,
    source_handoff_dir: Path,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    flip_audit_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    source_handoff = source_handoff_dir / HANDOFF_NAME
    source_ledger = source_handoff_dir / LEDGER_NAME
    source_contract = source_handoff_dir / CONTRACT_NAME
    if not source_handoff.exists():
        raise FileNotFoundError(source_handoff)
    handoff = pd.read_parquet(source_handoff)
    required_handoff = {"month", *CELL_COLUMNS}
    missing_handoff = sorted(required_handoff.difference(handoff.columns))
    if missing_handoff:
        raise ValueError(f"Source handoff missing required columns: {missing_handoff}")
    months = sorted(str(m) for m in handoff["month"].dropna().astype(str).unique())
    month_cells = _load_or_build_month_cells(
        flip_audit_dir=flip_audit_dir,
        baseline_smoke_dir=baseline_smoke_dir,
        promoted_smoke_dir=promoted_smoke_dir,
    )
    prior_rows = _prior_feature_rows(month_cells, months)
    stability_cols = [col for col in prior_rows.columns if col not in {"month", *CELL_COLUMNS}]
    collisions = [col for col in stability_cols if col in handoff.columns]
    if collisions:
        handoff = handoff.drop(columns=collisions)
    materialized = handoff.merge(prior_rows, on=["month", *CELL_COLUMNS], how="left", validate="many_to_one")
    for col in stability_cols:
        materialized[col] = pd.to_numeric(materialized[col], errors="coerce").astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_handoff = out_dir / HANDOFF_NAME
    materialized.to_parquet(out_handoff, index=False)
    copied_ledger = False
    if source_ledger.exists():
        shutil.copy2(source_ledger, out_dir / LEDGER_NAME)
        copied_ledger = True
    contract: dict[str, Any] = {}
    if source_contract.exists():
        try:
            contract = json.loads(source_contract.read_text())
        except Exception:
            contract = {"unparsed_source_contract": str(source_contract)}
    prior_any = materialized[stability_cols].notna().any(axis=1) if stability_cols else pd.Series(False, index=materialized.index)
    prior_history_cols = [col for col in stability_cols if col.endswith("_history_months")]
    no_prior_months = {}
    for month in months:
        month_rows = materialized["month"].astype(str).eq(month)
        if not prior_history_cols or not month_rows.any():
            no_prior_months[month] = None
        else:
            no_prior_months[month] = bool(materialized.loc[month_rows, prior_history_cols].fillna(0.0).sum(axis=1).eq(0.0).all())
    contract["cross_asset_stability_priors"] = {
        "status": "materialized",
        "source_handoff_dir": str(source_handoff_dir),
        "source_handoff_sha256": _sha256_file(source_handoff),
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "flip_audit_dir": str(flip_audit_dir),
        "month_cell_effects": str(flip_audit_dir / "promoted_cross_asset_month_cell_effects.csv"),
        "feature_count": int(len(stability_cols)),
        "feature_columns": stability_cols,
        "row_count": int(len(materialized)),
        "rows_with_any_prior": int(prior_any.sum()),
        "coverage_any_prior": float(prior_any.mean()) if len(materialized) else 0.0,
        "month_has_no_prior_history": no_prior_months,
        "source_ledger_copied": bool(copied_ledger),
        "leakage_contract": (
            "For each row, stability priors are computed only from OOF baseline-vs-promoted "
            "month-cell diagnostics with month strictly earlier than the row month. Rows with "
            "no prior history are left NaN; no current-month or future diagnostics are joined."
        ),
    }
    (out_dir / CONTRACT_NAME).write_text(json.dumps(_json_safe(contract), indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "materialize_cross_asset_stability_meta_handoff",
        "source_handoff_dir": str(source_handoff_dir),
        "out_dir": str(out_dir),
        "handoff_path": str(out_handoff),
        "contract_path": str(out_dir / CONTRACT_NAME),
        "ledger_path": str(out_dir / LEDGER_NAME) if copied_ledger else None,
        "row_count": int(len(materialized)),
        "input_column_count": int(len(handoff.columns)),
        "output_column_count": int(len(materialized.columns)),
        "stability_feature_count": int(len(stability_cols)),
        "rows_with_any_prior": int(prior_any.sum()),
        "coverage_any_prior": float(prior_any.mean()) if len(materialized) else 0.0,
        "month_has_no_prior_history": no_prior_months,
        "stability_columns": stability_cols,
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-handoff-dir", type=Path, default=DEFAULT_SOURCE_HANDOFF_DIR)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--flip-audit-dir", type=Path, default=DEFAULT_FLIP_AUDIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize(
        source_handoff_dir=args.source_handoff_dir,
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        flip_audit_dir=args.flip_audit_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe({"event": "cross_asset_stability_meta_handoff_materialized", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
