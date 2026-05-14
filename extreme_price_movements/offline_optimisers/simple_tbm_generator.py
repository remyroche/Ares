#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_DIR.parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from extreme_price_movements.offline_optimisers.params_store import (
    REPORTS_DIR,
    TBM_BEST_PARAMS_PER_CELL_CSV,
    TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV,
    TBM_GEOMETRY_GRID_CSV,
    load_inference_candidate_mask_params_per_bucket,
    market_report_path,
    normalize_market_mode,
)
from extreme_price_movements.strategy_registry import normalize_strategy_horizon

MARKET_MODE_SUFFIXES = {"spot": "_spot", "perps": "_perps"}


def _normalise_market_mode(*, perps: bool = False, market_mode: str | None = None) -> str:
    mode = str(market_mode or "").strip().lower()
    if mode in {"perp", "perps", "futures"} or perps:
        return "perps"
    return "spot"


def _mode_path(path: Path, market_mode: str) -> Path:
    mode = _normalise_market_mode(market_mode=market_mode)
    return path.with_name(f"{path.stem}_{mode}{path.suffix}")


def _existing_mode_input(path: Path, market_mode: str) -> Path:
    mode_path = _mode_path(path, market_mode)
    if mode_path.exists():
        return mode_path
    return path


def _normalize_horizon_token(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("H"):
        text = text[1:]
    try:
        return f"H{normalize_strategy_horizon(int(float(text)))}"
    except Exception:
        return "H5"


def _normalize_cell_key(cell_key: object) -> str:
    text = str(cell_key or "").strip()
    if not text:
        return text
    return re.sub(
        r"_H(\d+)$",
        lambda m: f"_H{normalize_strategy_horizon(int(m.group(1)))}",
        text,
    )


def _to_side_horizon(cell_key: object) -> str:
    parts = str(cell_key or "").split("_")
    if len(parts) >= 3 and parts[0] in {"MR", "TF"}:
        return f"{parts[1]}_{parts[2]}"
    return str(cell_key or "")


def _family_from_mode(mode: object) -> str:
    text = str(mode or "").lower()
    if "wide" in text:
        return "wide"
    return "tight"


def _rewrite_geometry_grid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell_key"] = out["cell_key"].map(_normalize_cell_key).map(_to_side_horizon)
    if "horizon" in out.columns:
        out["horizon"] = out["horizon"].map(_normalize_horizon_token)
    else:
        out["horizon"] = out["cell_key"].astype(str).str.extract(r"(_H\d+)$")[0].str[1:]
    out["family"] = out.get("family", out.get("mode", "")).map(_family_from_mode)
    dedup_cols = [
        c
        for c in [
            "cell_key",
            "family",
            "k_tp",
            "sl_as_tp_pct",
            "base_atr_window",
            "config_id",
        ]
        if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
    sort_cols = [c for c in ["cell_key", "family", "rank", "config_id"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _rewrite_best_params_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell_key"] = out["cell_key"].map(_normalize_cell_key).map(_to_side_horizon)
    out["horizon"] = out.get("horizon", out["cell_key"]).map(_normalize_horizon_token)
    dedup_cols = [
        c
        for c in [
            "cell_key",
            "config_id",
            "k_tp",
            "sl_as_tp_pct",
            "base_atr_window",
        ]
        if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
    sort_cols = [c for c in ["cell_key", "rank_in_cell", "config_id"] if c in out.columns]
    if "rank_in_cell" in out.columns:
        out["rank_in_cell"] = out.groupby("cell_key").cumcount() + 1
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _assert_family_coverage(grid_df: pd.DataFrame) -> None:
    fam = (
        grid_df.groupby("cell_key")["family"]
        .apply(lambda s: sorted(set(str(v) for v in s if pd.notna(v))))
        .to_dict()
    )
    missing = {
        cell: vals
        for cell, vals in fam.items()
        if not {"tight", "wide"}.issubset(set(vals))
    }
    if missing:
        raise ValueError(f"Missing tight/wide families for cells: {missing}")
def _build_side_horizon_best_params(cell_df: pd.DataFrame) -> pd.DataFrame:
    out = cell_df.copy()
    out["cell_key"] = out["cell_key"].map(_to_side_horizon)
    sort_cols = [c for c in ["cell_key", "rank_in_cell", "config_id"] if c in out.columns]
    out = out.sort_values(sort_cols)
    dedup_cols = [c for c in ["cell_key", "config_id", "k_tp", "sl_as_tp_pct", "base_atr_window"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="first")
    if "rank_in_cell" in out.columns:
        out["rank_in_cell"] = out.groupby("cell_key").cumcount() + 1
    return out.reset_index(drop=True)


def _write_inference_candidate_bucket_params(*, market_mode: str = "spot") -> Path:
    os.environ["EPM_MASK_STRATEGY_SKIP_REPORT_INPUTS"] = "1"
    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=20, market_mode=market_mode
    )
    out_path = market_report_path(
        REPORTS_DIR / "inference_candidate_mask_best_params_per_bucket.csv",
        market_mode,
    )
    if not strategies:
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path

    df = pd.DataFrame(strategies).copy()
    if "mask_params" in df.columns:
        df["mask_params_json"] = df["mask_params"].apply(
            lambda v: json.dumps(v, sort_keys=True) if isinstance(v, dict) else str(v)
        )
        df = df.drop(columns=["mask_params"])

    if {"trade_side", "source_horizon"}.issubset(df.columns):
        counts = df.groupby(["trade_side", "source_horizon"]).size()
        expected = {("long", 5), ("long", 10), ("short", 5), ("short", 10)}
        missing = expected.difference(set(counts.index.tolist()))
        if missing:
            raise ValueError(f"Missing top-5 bucket coverage for: {sorted(missing)}")
        if any(int(v) > 20 for v in counts.to_numpy()):
            raise ValueError(f"Expected at most 20 strategies per bucket, got counts={counts.to_dict()}")

    df = df.sort_values(["trade_side", "source_horizon", "adjusted_ranking_score"], ascending=[True, True, False]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return out_path


def _read_market_csv(path: Path, market_mode: str) -> tuple[pd.DataFrame, Path]:
    mode_path = market_report_path(path, market_mode)
    if not mode_path.exists():
        raise FileNotFoundError(mode_path)
    return pd.read_csv(mode_path), mode_path


def regenerate_simple_tbm_reports(
    *, perps: bool = False, market_mode: str | None = None
) -> tuple[Path, Path, Path]:
    market_mode = normalize_market_mode("perps" if perps else market_mode)
    grid_path = market_report_path(TBM_GEOMETRY_GRID_CSV, market_mode)
    cell_path = market_report_path(TBM_BEST_PARAMS_PER_CELL_CSV, market_mode)
    side_path = market_report_path(TBM_BEST_PARAMS_PER_SIDE_HORIZON_CSV, market_mode)

    grid_df, grid_read_path = _read_market_csv(TBM_GEOMETRY_GRID_CSV, market_mode)
    cell_df, cell_read_path = _read_market_csv(TBM_BEST_PARAMS_PER_CELL_CSV, market_mode)

    grid_df = _rewrite_geometry_grid(grid_df)
    cell_df = _rewrite_best_params_per_cell(cell_df)
    side_df = _build_side_horizon_best_params(cell_df)
    _assert_family_coverage(grid_df)
    grid_df["market_mode"] = market_mode
    cell_df["market_mode"] = market_mode
    side_df["market_mode"] = market_mode

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(grid_path, index=False)
    cell_df.to_csv(cell_path, index=False)
    side_df.to_csv(side_path, index=False)
    _write_inference_candidate_bucket_params(market_mode=market_mode)
    manifest = {
        "market_mode": market_mode,
        "input_geometry_grid": str(grid_read_path),
        "input_best_params_per_cell": str(cell_read_path),
    }
    if market_mode == "perps":
        manifest[
            "sl_liquidation_rule"
        ] = "stop_loss_pct <= liquidation_wall_pct / 3 using mark price where available"
    (REPORTS_DIR / f"tbm_{market_mode}_mode_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return grid_path, cell_path, side_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate simple TBM report tables")
    parser.add_argument(
        "--market-mode",
        choices=["spot", "perps"],
        default="spot",
        help="Market mode for TBM report files (default: spot).",
    )
    parser.add_argument(
        "--perps",
        action="store_true",
        help="Alias for --market-mode perps.",
    )
    args = parser.parse_args()
    grid_path, cell_path, side_path = regenerate_simple_tbm_reports(
        perps=args.perps, market_mode=args.market_mode
    )
    print(f"rewrote {grid_path}")
    print(f"rewrote {cell_path}")
    print(f"rewrote {side_path}")


if __name__ == "__main__":
    main()
