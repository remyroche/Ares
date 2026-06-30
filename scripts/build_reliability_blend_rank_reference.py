#!/usr/bin/env python3
"""Build frozen rank references for selected native reliability-blend scores."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.policy_rank_reference import (  # noqa: E402
    persist_fullscope_score_distribution_reference,
)
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import (  # noqa: E402
    STRATEGY_IDS,
    _file_sha256,
    _json_safe,
    _load_default_variants,
)


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k"
)
DEFAULT_CONFIG = Path("config/reliability_blend_default_configs.json")


def _selected_score_frames(
    scores: pd.DataFrame,
    variants: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, Any]] = []
    for head, strategy_id in STRATEGY_IDS.items():
        variant = variants.get(head)
        col = f"blend_{variant}_score" if variant else ""
        group = scores.loc[scores["head"].astype(str).eq(head)].copy()
        if not variant or col not in group.columns or group.empty:
            diagnostics.append(
                {
                    "head": head,
                    "strategy_id": strategy_id,
                    "variant": variant,
                    "score_col": col,
                    "status": "missing",
                    "rows": int(len(group)),
                }
            )
            continue
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(group["timestamp"], utc=True, errors="coerce"),
                "symbol": group["symbol"].astype(str),
                "calibrated_score": pd.to_numeric(group[col], errors="coerce"),
                "head": head,
            }
        )
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["calibrated_score"])
        frames[strategy_id] = out
        diagnostics.append(
            {
                "head": head,
                "strategy_id": strategy_id,
                "variant": variant,
                "score_col": col,
                "status": "ok",
                "rows": int(len(out)),
                "timestamp_min": out["timestamp"].min().isoformat() if len(out) else None,
                "timestamp_max": out["timestamp"].max().isoformat() if len(out) else None,
            }
        )
    return frames, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--component-scores", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--run-id", type=str, default="reliability_blend_native_rank_reference_20260624")
    parser.add_argument("--market-mode", type=str, default="perps")
    args = parser.parse_args()

    score_path = args.component_scores or args.source_dir / "reliability_blend_component_scores.parquet"
    scores = pd.read_parquet(score_path)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    variants = _load_default_variants(args.config)
    frames, diagnostics = _selected_score_frames(scores, variants)
    if not frames:
        raise RuntimeError("No selected reliability-blend score frames were available.")
    manifest_path = persist_fullscope_score_distribution_reference(
        frames,
        data_root=args.data_root,
        run_id=args.run_id,
        market_mode=args.market_mode,
        score_col="calibrated_score",
        provenance={
            "source": "selected_native_reliability_blend_component_scores",
            "component_scores": str(score_path),
            "component_scores_sha256": _file_sha256(score_path),
            "config": str(args.config),
            "config_sha256": _file_sha256(args.config),
            "default_variants": variants,
        },
    )
    freeze_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "build_reliability_blend_rank_reference",
        "run_id": str(args.run_id),
        "market_mode": str(args.market_mode),
        "score_path": str(score_path),
        "score_sha256": _file_sha256(score_path),
        "config": str(args.config),
        "config_sha256": _file_sha256(args.config),
        "rank_reference_manifest": str(manifest_path),
        "score_column": "calibrated_score",
        "score_contract": "selected native reliability blend scores; percentile mapping only",
        "diagnostics": diagnostics,
    }
    out_path = Path(args.data_root) / "artifacts" / str(args.run_id) / "reliability_blend_rank_reference_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(freeze_manifest), indent=2) + "\n")
    print(json.dumps(_json_safe(freeze_manifest), indent=2)[:6000])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
