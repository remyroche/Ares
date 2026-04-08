#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.config import CFG
from extreme_price_movements.path_utils import resolve_reports_dir
from extreme_price_movements.run_pipeline import (
    _configure_report_roots,
    _label_artifacts_ready,
    _normalize_cfg_paths,
    _resolve_ts_sig,
)
from extreme_price_movements.strategy_registry import (
    get_strategies,
    strategy_runtime_horizons,
)


@dataclass
class DatasetCheck:
    dataset: str
    path: str
    rows: int
    feature_cols: int
    y_pos_rate: float
    tp_rate: float
    timeout_rate: float
    sl_rate: float
    positive_weight_rate: float
    finite_return_rate: float
    unique_symbols: int
    warnings: list[str]
    errors: list[str]


def _fmt_pct(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{100.0 * float(v):.2f}%"


def _required_alpha_datasets(cfg: dict) -> tuple[list[str], dict[str, dict[str, str]]]:
    required: list[str] = []
    variant_map: dict[str, dict[str, str]] = {}
    variants = [
        str(v)
        for v in cfg.get("base_geometry_archetypes", ["tight", "balanced", "wide"])
        if str(v) != "balanced"
    ]
    for strat in get_strategies(cfg):
        strategy_id = str(strat["strategy_id"])
        for horizon in strategy_runtime_horizons(strat, cfg):
            base_key = f"train_{strategy_id}_{int(horizon)}"
            required.append(base_key)
            variant_map[base_key] = {}
            if bool(cfg.get("base_geometry_train_variants", True)):
                for variant in variants:
                    variant_key = f"{base_key}_{variant}"
                    required.append(variant_key)
                    variant_map[base_key][variant] = variant_key
    return required, variant_map


def _scan_feature_health(path: str, schema_names: list[str]) -> tuple[int, float]:
    meta_cols = {
        "__y_bin__",
        "__y_ret__",
        "__y_outcome__",
        "__w__",
        "__ts__",
        "__symbol__",
        "ts",
        "symbol",
    }
    feature_cols = [c for c in schema_names if c not in meta_cols]
    if not feature_cols:
        return 0, 0.0
    table = pq.read_table(path, columns=feature_cols)
    if table.num_rows == 0:
        return len(feature_cols), 0.0
    non_null_cols = 0
    for col in table.itercolumns():
        if col.null_count < len(col):
            non_null_cols += 1
    return len(feature_cols), non_null_cols / max(len(feature_cols), 1)


def _verify_dataset(path: str, dataset: str) -> DatasetCheck:
    errors: list[str] = []
    warnings: list[str] = []
    pf = pq.ParquetFile(path)
    rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
    schema_names = list(pf.schema.names)
    required_cols = {
        "__y_bin__",
        "__y_ret__",
        "__y_outcome__",
        "__w__",
        "__ts__",
        "__symbol__",
    }
    missing = sorted(required_cols - set(schema_names))
    if missing:
        errors.append(f"missing columns: {missing}")

    feature_cols, non_null_feature_ratio = _scan_feature_health(path, schema_names)
    if feature_cols == 0:
        errors.append("no feature columns")
    elif non_null_feature_ratio < 0.50:
        errors.append(
            f"too many null feature columns: non_null_feature_ratio={non_null_feature_ratio:.3f}"
        )

    cols_to_read = sorted(required_cols & set(schema_names))
    table = pq.read_table(path, columns=cols_to_read) if cols_to_read else None
    df = table.to_pandas() if table is not None else pd.DataFrame()

    if rows <= 0:
        errors.append("empty dataset")
    if not df.empty:
        y_bin = pd.to_numeric(df["__y_bin__"], errors="coerce").astype(np.float32)
        y_ret = pd.to_numeric(df["__y_ret__"], errors="coerce").astype(np.float32)
        y_out = pd.to_numeric(df["__y_outcome__"], errors="coerce").astype(np.float32)
        w = pd.to_numeric(df["__w__"], errors="coerce").astype(np.float32)
        ts = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
        sym = df["__symbol__"].astype(str)

        bad_y_bin = ~np.isin(y_bin.dropna().to_numpy(), np.array([0.0, 1.0], dtype=np.float32))
        if bad_y_bin.any():
            errors.append("invalid __y_bin__ values outside {0,1}")

        bad_y_out = ~np.isin(
            y_out.dropna().to_numpy(), np.array([0.0, 1.0, 2.0], dtype=np.float32)
        )
        if bad_y_out.any():
            errors.append("invalid __y_outcome__ values outside {0,1,2}")

        if y_bin.isna().any():
            errors.append("NaN labels in __y_bin__")
        if y_out.isna().any():
            errors.append("NaN labels in __y_outcome__")
        if not np.isfinite(y_ret.to_numpy()).all():
            errors.append("non-finite values in __y_ret__")
        if not np.isfinite(w.to_numpy()).all():
            errors.append("non-finite values in __w__")
        if (w < 0).any():
            errors.append("negative sample weights")
        if float(np.sum(w > 0)) <= 0:
            errors.append("all sample weights are zero")
        if ts.isna().any():
            errors.append("NaT values in __ts__")
        if sym.isna().any() or (sym.str.len() == 0).any():
            errors.append("empty symbols in __symbol__")

        key_dupes = pd.DataFrame({"ts": ts, "symbol": sym}).duplicated().sum()
        if int(key_dupes) > 0:
            errors.append(f"duplicate (__ts__, __symbol__) keys: {int(key_dupes)}")

        y_bin_np = y_bin.to_numpy(dtype=np.float32, copy=False)
        outcome_np = y_out.to_numpy(dtype=np.float32, copy=False)
        outcome_consistent = y_bin_np == (outcome_np == 2.0).astype(np.float32)
        mismatch_rate = 1.0 - float(np.mean(outcome_consistent))
        if mismatch_rate > 0.0:
            errors.append(f"__y_bin__/__y_outcome__ mismatch_rate={mismatch_rate:.6f}")

        unique_y = np.unique(y_bin_np)
        if len(unique_y) < 2:
            errors.append("degenerate binary target: only one class present")

        tp_mask = outcome_np == 2.0
        sl_mask = outcome_np == 0.0
        to_mask = outcome_np == 1.0
        tp_rate = float(np.mean(tp_mask))
        sl_rate = float(np.mean(sl_mask))
        timeout_rate = float(np.mean(to_mask))
        if abs((tp_rate + sl_rate + timeout_rate) - 1.0) > 1e-4:
            errors.append("outcome rates do not sum to 1")

        if tp_mask.any() and sl_mask.any():
            tp_ret_med = float(np.nanmedian(y_ret.to_numpy()[tp_mask]))
            sl_ret_med = float(np.nanmedian(y_ret.to_numpy()[sl_mask]))
            if not (tp_ret_med > sl_ret_med):
                warnings.append(
                    f"unexpected return ordering: median_tp_ret={tp_ret_med:.6f} median_sl_ret={sl_ret_med:.6f}"
                )

        positive_weight_rate = float(np.mean(w.to_numpy() > 0))
        finite_return_rate = float(np.mean(np.isfinite(y_ret.to_numpy())))
        unique_symbols = int(sym.nunique())
        if unique_symbols < 5:
            warnings.append(f"very low symbol coverage: {unique_symbols}")
        if rows < 1000:
            warnings.append(f"very low row count: {rows}")
    else:
        tp_rate = float("nan")
        sl_rate = float("nan")
        timeout_rate = float("nan")
        positive_weight_rate = 0.0
        finite_return_rate = 0.0
        unique_symbols = 0

    return DatasetCheck(
        dataset=dataset,
        path=path,
        rows=rows,
        feature_cols=feature_cols,
        y_pos_rate=float(np.mean(df["__y_bin__"])) if "__y_bin__" in df.columns and len(df) else float("nan"),
        tp_rate=tp_rate,
        timeout_rate=timeout_rate,
        sl_rate=sl_rate,
        positive_weight_rate=positive_weight_rate,
        finite_return_rate=finite_return_rate,
        unique_symbols=unique_symbols,
        warnings=warnings,
        errors=errors,
    )


def main() -> int:
    cfg = dict(CFG)
    _normalize_cfg_paths(cfg)
    _configure_report_roots(cfg)
    ts_sig = _resolve_ts_sig(cfg, None)
    if ts_sig is None:
        print("[verify_labels] no timestamp signature resolved")
        return 2
    if not _label_artifacts_ready(cfg, ts_sig):
        print("[verify_labels] labels not ready yet")
        return 2

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    labels_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "labels")
    required, variant_map = _required_alpha_datasets(cfg)

    checks: list[DatasetCheck] = []
    fatal_errors: list[str] = []
    for dataset in required:
        path = os.path.join(labels_dir, f"{dataset}.parquet")
        if not os.path.exists(path):
            fatal_errors.append(f"{dataset}: missing parquet")
            continue
        checks.append(_verify_dataset(path, dataset))

    check_by_name = {c.dataset: c for c in checks}
    for base_key, variants in variant_map.items():
        if not variants:
            continue
        base = check_by_name.get(base_key)
        if base is None:
            fatal_errors.append(f"{base_key}: missing base dataset for variant reconciliation")
            continue
        variant_rows = 0
        for variant_key in variants.values():
            variant_check = check_by_name.get(variant_key)
            if variant_check is None:
                fatal_errors.append(f"{base_key}: missing variant {variant_key}")
                continue
            variant_rows += int(variant_check.rows)
        if variant_rows and int(base.rows) != int(variant_rows):
            fatal_errors.append(
                f"{base_key}: base rows={base.rows} but variants sum to {variant_rows}"
            )

    for check in checks:
        fatal_errors.extend(f"{check.dataset}: {err}" for err in check.errors)

    reports_dir = resolve_reports_dir(cfg.get("reports_root")) / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for check in checks:
        summary_rows.append(
            {
                "dataset": check.dataset,
                "rows": check.rows,
                "feature_cols": check.feature_cols,
                "y_pos_rate": check.y_pos_rate,
                "tp_rate": check.tp_rate,
                "timeout_rate": check.timeout_rate,
                "sl_rate": check.sl_rate,
                "positive_weight_rate": check.positive_weight_rate,
                "finite_return_rate": check.finite_return_rate,
                "unique_symbols": check.unique_symbols,
                "warnings": " | ".join(check.warnings),
                "errors": " | ".join(check.errors),
                "path": check.path,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    summary_df.to_csv(reports_dir / "label_verification.csv", index=False)

    md_lines = [
        f"# Label Verification Report — {run_id}",
        "",
        f"- Checked datasets: {len(required)}",
        f"- Fatal errors: {len(fatal_errors)}",
        f"- Datasets with warnings: {sum(bool(c.warnings) for c in checks)}",
        "",
        "| Dataset | Rows | Feats | y=1 | TP | TO | SL | +w | Finite ret | Symbols |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for check in sorted(checks, key=lambda x: x.dataset):
        md_lines.append(
            "| "
            + " | ".join(
                [
                    check.dataset,
                    f"{check.rows:,}",
                    str(check.feature_cols),
                    _fmt_pct(check.y_pos_rate),
                    _fmt_pct(check.tp_rate),
                    _fmt_pct(check.timeout_rate),
                    _fmt_pct(check.sl_rate),
                    _fmt_pct(check.positive_weight_rate),
                    _fmt_pct(check.finite_return_rate),
                    str(check.unique_symbols),
                ]
            )
            + " |"
        )
    if fatal_errors:
        md_lines.extend(["", "## Fatal Errors"])
        md_lines.extend([f"- {err}" for err in fatal_errors])
    warn_items = [f"{c.dataset}: {w}" for c in checks for w in c.warnings]
    if warn_items:
        md_lines.extend(["", "## Warnings"])
        md_lines.extend([f"- {item}" for item in warn_items])
    (reports_dir / "label_verification.md").write_text("\n".join(md_lines), encoding="utf-8")
    (reports_dir / "label_verification.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "checked_datasets": len(required),
                "fatal_errors": fatal_errors,
                "warnings": warn_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if fatal_errors:
        print("[verify_labels] FAILED")
        for err in fatal_errors:
            print(f"[verify_labels] {err}")
        print(f"[verify_labels] report={reports_dir / 'label_verification.md'}")
        return 1

    print("[verify_labels] PASSED")
    print(f"[verify_labels] report={reports_dir / 'label_verification.md'}")
    for check in sorted(checks, key=lambda x: x.dataset):
        print(
            "[verify_labels] "
            f"{check.dataset}: rows={check.rows:,} feats={check.feature_cols} "
            f"y1={_fmt_pct(check.y_pos_rate)} tp={_fmt_pct(check.tp_rate)} "
            f"to={_fmt_pct(check.timeout_rate)} sl={_fmt_pct(check.sl_rate)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
