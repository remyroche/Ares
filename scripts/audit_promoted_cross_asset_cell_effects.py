#!/usr/bin/env python3
"""Cell-aware audit for promoted cross-asset meta features.

The global promoted-handoff audit is necessary but not sufficient for the
cross-asset archetype plan.  This script compares baseline and promoted meta
smoke predictions inside side x archetype cells and reports whether the
promoted representation helps at least one supported cell without causing
catastrophic damage in major cells.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_xmarket_v1"
)
DEFAULT_BASELINE_SMOKE_DIR = DEFAULT_ROOT / "train_meta_smoke_baseline_for_promoted_compare_v2"
DEFAULT_PROMOTED_HANDOFF_DIR = DEFAULT_ROOT / "train_meta_handoff_promoted_cross_asset_v1"
DEFAULT_PROMOTED_SMOKE_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "train_meta_smoke_v2"
DEFAULT_OUT_DIR = DEFAULT_PROMOTED_HANDOFF_DIR / "promoted_cross_asset_cell_effect_audit_v1"

PREDICTIONS_NAME = "s52_train_meta_regime_handoff_smoke_predictions.parquet"
MANIFEST_NAME = "manifest.json"
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
GROUP_COLUMNS = ("side_name", "source_semantic_family")
KEEP_FRACS = (0.10, 0.20, 0.30)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _score_column(selector: str) -> str:
    selector = str(selector or "").strip()
    if selector == "base_score":
        return "score_base"
    if selector.startswith("score_"):
        return selector
    if selector.startswith("meta_"):
        return f"score_{selector}"
    return selector


def _best_score_column(smoke_dir: Path) -> tuple[str, str]:
    manifest = _read_json(smoke_dir / MANIFEST_NAME)
    selector = str((manifest.get("best_selector") or {}).get("selector") or "base_score")
    return selector, _score_column(selector)


def _num(values: Any, *, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    if values is None:
        if index is None:
            return pd.Series(dtype=np.float32)
        return pd.Series(default, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _rate(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) == 0:
        return float("nan")
    return float(arr.clip(0.0, 1.0).mean())


def _mean(values: Any) -> float:
    arr = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())


def _support_metrics(cell: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(cell.get("__ts__"), utc=True, errors="coerce")
    weeks = ts.dt.strftime("%G-W%V").fillna("unknown")
    symbols = cell.get("__symbol__", pd.Series("unknown", index=cell.index)).astype(str)
    return {
        "cell_rows": int(len(cell)),
        "month_count": int(cell.get("month", pd.Series(dtype=str)).astype(str).nunique()),
        "symbol_count": int(symbols.nunique()),
        "clean_rows": int(_num(cell.get("clean_exec"), index=cell.index, default=0.0).fillna(0.0).gt(0.5).sum()),
        "positive_exec_rows": int(_num(cell.get("exec_margin"), index=cell.index, default=np.nan).gt(0.0).sum()),
        "max_single_asset_share": float(symbols.value_counts(normalize=True).iloc[0]) if len(symbols) else float("nan"),
        "max_single_week_share": float(weeks.value_counts(normalize=True).iloc[0]) if len(weeks) else float("nan"),
    }


def _top_cell_metrics(cell: pd.DataFrame, score_col: str, keep_frac: float) -> dict[str, Any]:
    if score_col not in cell.columns:
        raise ValueError(f"Missing score column {score_col!r}")
    scored = cell.copy()
    scored["_score__tmp"] = _num(scored.get(score_col), index=scored.index)
    scored = scored[scored["_score__tmp"].notna()]
    if scored.empty:
        return {
            "selected_rows": 0,
            "exec_margin": float("nan"),
            "clean_exec_precision": float("nan"),
            "full_path_bad_mae": float("nan"),
            "timeout": float("nan"),
            "dirty_positive": float("nan"),
            "mfe_before_mae": float("nan"),
            "mae_before_mfe": float("nan"),
            "underwater_bars": float("nan"),
            "cell_oracle_overlap": float("nan"),
        }
    n = max(1, int(math.ceil(len(scored) * float(keep_frac))))
    top = scored.sort_values("_score__tmp", ascending=False).head(n)
    oracle = scored.sort_values("exec_margin", ascending=False).head(n) if "exec_margin" in scored.columns else scored.head(0)
    top_keys = set(map(tuple, top[list(KEY_COLUMNS)].astype(str).to_numpy())) if all(col in top.columns for col in KEY_COLUMNS) else set()
    oracle_keys = (
        set(map(tuple, oracle[list(KEY_COLUMNS)].astype(str).to_numpy()))
        if all(col in oracle.columns for col in KEY_COLUMNS)
        else set()
    )
    overlap = float(len(top_keys & oracle_keys) / max(1, len(oracle_keys))) if oracle_keys else float("nan")
    return {
        "selected_rows": int(len(top)),
        "exec_margin": _mean(top.get("exec_margin")),
        "clean_exec_precision": _rate(top.get("clean_exec")),
        "full_path_bad_mae": _rate(top.get("full_path_bad_mae_1r")),
        "timeout": _rate(top.get("timeout")),
        "dirty_positive": _rate(top.get("dirty_positive")),
        "mfe_before_mae": _rate(top.get("mfe_before_mae_1r")),
        "mae_before_mfe": _rate(top.get("mae_before_mfe_1r")),
        "underwater_bars": _mean(top.get("underwater_bars_before_mfe_1r")),
        "cell_oracle_overlap": overlap,
    }


def _cell_rows(
    baseline: pd.DataFrame,
    promoted: pd.DataFrame,
    *,
    baseline_score_col: str,
    promoted_score_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_key = list(GROUP_COLUMNS)
    for key, base_cell in baseline.groupby(group_key, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        mask = pd.Series(True, index=promoted.index)
        for col, value in zip(group_key, key, strict=False):
            mask &= promoted[col].astype(str).eq(str(value))
        promoted_cell = promoted.loc[mask].copy()
        if promoted_cell.empty:
            continue
        support = _support_metrics(base_cell)
        for keep_frac in KEEP_FRACS:
            base_metrics = _top_cell_metrics(base_cell, baseline_score_col, keep_frac)
            promoted_metrics = _top_cell_metrics(promoted_cell, promoted_score_col, keep_frac)
            record: dict[str, Any] = {
                "keep_frac": float(keep_frac),
                **{col: value for col, value in zip(group_key, key, strict=False)},
                **support,
            }
            for metric, value in base_metrics.items():
                record[f"baseline_{metric}"] = value
            for metric, value in promoted_metrics.items():
                record[f"promoted_{metric}"] = value
            for metric in (
                "exec_margin",
                "clean_exec_precision",
                "full_path_bad_mae",
                "timeout",
                "dirty_positive",
                "mfe_before_mae",
                "mae_before_mfe",
                "underwater_bars",
                "cell_oracle_overlap",
            ):
                record[f"delta_{metric}"] = promoted_metrics[metric] - base_metrics[metric]
            rows.append(record)
    return pd.DataFrame(rows)


def _classify_cells(
    cells: pd.DataFrame,
    *,
    min_valid_rows: int,
    min_months: int,
    min_clean_rows: int,
    min_positive_rows: int,
    max_asset_share: float,
    max_week_share: float,
) -> pd.DataFrame:
    if cells.empty:
        return cells
    out = cells.copy()
    out["support_pass"] = (
        out["cell_rows"].ge(int(min_valid_rows))
        & out["month_count"].ge(int(min_months))
        & out["clean_rows"].ge(int(min_clean_rows))
        & out["positive_exec_rows"].ge(int(min_positive_rows))
        & out["max_single_asset_share"].le(float(max_asset_share))
        & out["max_single_week_share"].le(float(max_week_share))
    )
    out["cell_value_score"] = (
        (out["delta_exec_margin"].fillna(0.0) / 0.002).clip(-2.0, 2.0)
        + out["delta_clean_exec_precision"].fillna(0.0)
        - out["delta_full_path_bad_mae"].fillna(0.0)
        - 0.50 * out["delta_timeout"].fillna(0.0)
        + 0.50 * out["delta_mfe_before_mae"].fillna(0.0)
        - 0.50 * out["delta_mae_before_mfe"].fillna(0.0)
        + 0.25 * out["delta_cell_oracle_overlap"].fillna(0.0)
    )
    out["beneficial_supported_cell"] = (
        out["support_pass"]
        & (
            out["delta_exec_margin"].gt(0.0005)
            | out["delta_clean_exec_precision"].gt(0.03)
            | out["delta_full_path_bad_mae"].lt(-0.05)
            | out["delta_mfe_before_mae"].gt(0.05)
            | out["delta_cell_oracle_overlap"].gt(0.03)
        )
        & out["delta_exec_margin"].ge(-0.0015)
        & out["delta_full_path_bad_mae"].le(0.05)
        & out["delta_timeout"].le(0.02)
    )
    out["catastrophic_supported_degradation"] = (
        out["support_pass"]
        & (
            out["delta_exec_margin"].lt(-0.0020)
            | out["delta_full_path_bad_mae"].gt(0.08)
            | out["delta_timeout"].gt(0.03)
            | out["delta_clean_exec_precision"].lt(-0.08)
        )
    )
    return out.sort_values(["keep_frac", "support_pass", "cell_value_score"], ascending=[True, False, False])


def _summary(cells: pd.DataFrame) -> dict[str, Any]:
    if cells.empty:
        return {
            "status": "no_cells",
            "supported_cells": 0,
            "beneficial_supported_cells": 0,
            "catastrophic_supported_degradation_cells": 0,
        }
    supported = cells[cells["support_pass"]]
    beneficial = cells[cells["beneficial_supported_cell"]]
    damaged = cells[cells["catastrophic_supported_degradation"]]
    keep10 = cells[cells["keep_frac"].eq(0.10)]
    status = "pass" if len(beneficial) >= 1 and len(damaged) == 0 else "diagnostic_or_blocked"
    return {
        "status": status,
        "cell_effect_status": (
            "supported_cell_lift_without_major_damage" if status == "pass" else "requires_deeper_meta_or_repair"
        ),
        "cells": int(len(cells)),
        "supported_cells": int(len(supported)),
        "beneficial_supported_cells": int(len(beneficial)),
        "catastrophic_supported_degradation_cells": int(len(damaged)),
        "keep10_supported_cells": int(keep10["support_pass"].sum()) if not keep10.empty else 0,
        "keep10_beneficial_supported_cells": int(keep10["beneficial_supported_cell"].sum()) if not keep10.empty else 0,
        "keep10_catastrophic_supported_degradation_cells": int(keep10["catastrophic_supported_degradation"].sum())
        if not keep10.empty
        else 0,
        "best_supported_cells": _json_safe(
            supported.sort_values("cell_value_score", ascending=False)
            .head(10)[
                [
                    "keep_frac",
                    "side_name",
                    "source_semantic_family",
                    "cell_rows",
                    "cell_value_score",
                    "delta_exec_margin",
                    "delta_clean_exec_precision",
                    "delta_full_path_bad_mae",
                    "delta_timeout",
                    "delta_mfe_before_mae",
                    "delta_cell_oracle_overlap",
                    "beneficial_supported_cell",
                ]
            ]
            .to_dict("records")
        ),
        "damaged_supported_cells": _json_safe(
            damaged.sort_values("cell_value_score")
            .head(10)[
                [
                    "keep_frac",
                    "side_name",
                    "source_semantic_family",
                    "cell_rows",
                    "cell_value_score",
                    "delta_exec_margin",
                    "delta_clean_exec_precision",
                    "delta_full_path_bad_mae",
                    "delta_timeout",
                    "delta_mfe_before_mae",
                    "delta_cell_oracle_overlap",
                ]
            ]
            .to_dict("records")
        ),
    }


def _write_markdown(out_dir: Path, manifest: dict[str, Any], cells: pd.DataFrame) -> Path:
    summary = manifest["summary"]
    best = pd.DataFrame(summary.get("best_supported_cells") or [])
    damaged = pd.DataFrame(summary.get("damaged_supported_cells") or [])
    lines = [
        "# Promoted Cross-Asset Cell Effect Audit",
        "",
        "## Verdict",
        "",
        f"- status: `{summary.get('status')}`",
        f"- cell effect status: `{summary.get('cell_effect_status')}`",
        f"- supported cells: `{summary.get('supported_cells')}`",
        f"- beneficial supported cells: `{summary.get('beneficial_supported_cells')}`",
        f"- damaged supported cells: `{summary.get('catastrophic_supported_degradation_cells')}`",
        "",
        "## Scope",
        "",
        f"- baseline selector: `{manifest.get('baseline_selector')}`",
        f"- promoted selector: `{manifest.get('promoted_selector')}`",
        f"- grouping: `{'+'.join(GROUP_COLUMNS)}`",
        f"- keep fractions: `{', '.join(str(x) for x in KEEP_FRACS)}`",
        "",
        "## Best Supported Cells",
        "",
    ]
    lines.append(best.to_markdown(index=False) if not best.empty else "_No supported cells._")
    lines.extend(["", "## Damaged Supported Cells", ""])
    lines.append(damaged.to_markdown(index=False) if not damaged.empty else "_No catastrophic supported degradation cells._")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an OOF smoke-level cell audit. Passing here supports deeper train_meta evaluation; it is not frozen replay evidence.",
            "A block here means the promoted representation is still useful as context, but the meta layer must learn when to use it rather than applying it as a broad policy.",
        ]
    )
    path = out_dir / "promoted_cross_asset_cell_effect_audit.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_audit(
    *,
    baseline_smoke_dir: Path,
    promoted_smoke_dir: Path,
    out_dir: Path,
    min_valid_rows: int = 30,
    min_months: int = 2,
    min_clean_rows: int = 5,
    min_positive_rows: int = 5,
    max_asset_share: float = 0.80,
    max_week_share: float = 0.80,
) -> dict[str, Any]:
    baseline_selector, baseline_score_col = _best_score_column(baseline_smoke_dir)
    promoted_selector, promoted_score_col = _best_score_column(promoted_smoke_dir)
    baseline = pd.read_parquet(baseline_smoke_dir / PREDICTIONS_NAME)
    promoted = pd.read_parquet(promoted_smoke_dir / PREDICTIONS_NAME)
    for frame_name, frame, score_col in (
        ("baseline", baseline, baseline_score_col),
        ("promoted", promoted, promoted_score_col),
    ):
        missing = [col for col in (*GROUP_COLUMNS, *KEY_COLUMNS, score_col) if col not in frame.columns]
        if missing:
            raise ValueError(f"{frame_name} predictions missing required columns: {missing}")
    cells = _cell_rows(
        baseline,
        promoted,
        baseline_score_col=baseline_score_col,
        promoted_score_col=promoted_score_col,
    )
    cells = _classify_cells(
        cells,
        min_valid_rows=min_valid_rows,
        min_months=min_months,
        min_clean_rows=min_clean_rows,
        min_positive_rows=min_positive_rows,
        max_asset_share=max_asset_share,
        max_week_share=max_week_share,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_path = out_dir / "promoted_cross_asset_cell_effects.csv"
    cells.to_csv(cell_path, index=False)
    manifest = {
        "generated_by": "audit_promoted_cross_asset_cell_effects",
        "baseline_smoke_dir": str(baseline_smoke_dir),
        "promoted_smoke_dir": str(promoted_smoke_dir),
        "baseline_selector": baseline_selector,
        "baseline_score_col": baseline_score_col,
        "promoted_selector": promoted_selector,
        "promoted_score_col": promoted_score_col,
        "support_rule": {
            "min_valid_rows": int(min_valid_rows),
            "min_months": int(min_months),
            "min_clean_rows": int(min_clean_rows),
            "min_positive_rows": int(min_positive_rows),
            "max_asset_share": float(max_asset_share),
            "max_week_share": float(max_week_share),
        },
        "leakage_contract": {
            "prediction_source": "month-forward OOF smoke predictions from baseline and promoted train_meta handoffs",
            "labels_used_for": "offline diagnostics only",
            "selection": "top-k within each side x source_semantic_family cell using each smoke's OOF score",
        },
        "summary": _summary(cells),
        "outputs": {
            "cell_effects": str(cell_path),
            "json": str(out_dir / "promoted_cross_asset_cell_effect_audit.json"),
            "markdown": str(out_dir / "promoted_cross_asset_cell_effect_audit.md"),
        },
    }
    markdown = _write_markdown(out_dir, manifest, cells)
    manifest["outputs"]["markdown"] = str(markdown)
    (out_dir / "promoted_cross_asset_cell_effect_audit.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True)
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-smoke-dir", type=Path, default=DEFAULT_BASELINE_SMOKE_DIR)
    parser.add_argument("--promoted-smoke-dir", type=Path, default=DEFAULT_PROMOTED_SMOKE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-valid-rows", type=int, default=30)
    parser.add_argument("--min-months", type=int, default=2)
    parser.add_argument("--min-clean-rows", type=int, default=5)
    parser.add_argument("--min-positive-rows", type=int, default=5)
    parser.add_argument("--max-asset-share", type=float, default=0.80)
    parser.add_argument("--max-week-share", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(
        baseline_smoke_dir=args.baseline_smoke_dir,
        promoted_smoke_dir=args.promoted_smoke_dir,
        out_dir=args.out_dir,
        min_valid_rows=args.min_valid_rows,
        min_months=args.min_months,
        min_clean_rows=args.min_clean_rows,
        min_positive_rows=args.min_positive_rows,
        max_asset_share=args.max_asset_share,
        max_week_share=args.max_week_share,
    )
    print(json.dumps(_json_safe({"event": "promoted_cross_asset_cell_effect_audit_done", **manifest}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
