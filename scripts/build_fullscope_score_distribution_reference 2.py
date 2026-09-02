#!/usr/bin/env python3
"""Build a full-scope score CDF reference for percentile mapping only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.inference.parity import strategy_core_id, strategy_side
from extreme_price_movements.inference.policy_rank_reference import (
    persist_fullscope_score_distribution_reference,
)


def _strategy_id_from_meta_oof_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("meta_oof_"):
        stem = stem[len("meta_oof_") :]
    for suffix in ("_tbm_clf", "_correctness_clf", "_clf", "_reg"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_correctness"):
        stem = stem[: -len("_correctness")]
    return stem


def _is_route_overlay_path(path: Path) -> bool:
    stem = path.stem
    return bool(re.search(r"_(mr|tf)_tbm_clf$", stem) or re.search(r"_(mr|tf)_clf$", stem))


def _score_column(frame: pd.DataFrame) -> str:
    for col in ("oof_meta_clf", "oof_pred", "clf", "calibrated_score"):
        if col in frame.columns:
            return col
    raise ValueError("meta OOF frame has no usable score column")


def _load_model_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "models" / "model_state_meta.manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_reference(*, data_root: Path, run_id: str, market_mode: str) -> Path:
    run_root = data_root / "artifacts" / run_id
    meta_dir = run_root / "meta_oof"
    if not meta_dir.exists():
        raise FileNotFoundError(f"missing meta_oof directory: {meta_dir}")

    strategy_frames: dict[str, pd.DataFrame] = {}
    score_sources: dict[str, str] = {}
    for path in sorted(meta_dir.glob("meta_oof_*_clf.parquet")):
        if _is_route_overlay_path(path):
            continue
        sid = _strategy_id_from_meta_oof_path(path)
        if not sid:
            continue
        frame = pd.read_parquet(path)
        src_score_col = _score_column(frame)
        out = pd.DataFrame(
            {
                "calibrated_score": pd.to_numeric(
                    frame[src_score_col], errors="coerce"
                ),
                "strategy_id": sid,
            }
        )
        for col in ("timestamp", "symbol", "side"):
            if col in frame.columns:
                out[col] = frame[col]
        if "side" not in out.columns:
            side = strategy_side(sid)
            out["side"] = side if side else ""
        out["strategy_core_id"] = strategy_core_id(sid)
        strategy_frames[sid] = out
        score_sources[sid] = str(path.relative_to(run_root)) + f":{src_score_col}"

    if not strategy_frames:
        raise ValueError(f"no meta_oof score frames found under {meta_dir}")

    model_manifest = _load_model_manifest(run_root)
    provenance = {
        "source": "fullscope_meta_oof_score_distribution",
        "score_semantics": (
            "simple_policy_optimiser deployment score; no simple_position_sizer "
            "calibration is applied"
        ),
        "source_score_columns": score_sources,
        "model_state_meta_manifest": "models/model_state_meta.manifest.json",
        "source_model_fit_start": model_manifest.get("source_model_fit_start"),
        "source_model_fit_end": model_manifest.get("source_model_fit_end"),
        "source_role": model_manifest.get("source_role"),
        "is_oof_within_fullscope_fit_period": True,
    }
    return persist_fullscope_score_distribution_reference(
        strategy_frames,
        data_root=data_root,
        run_id=run_id,
        market_mode=market_mode,
        score_col="calibrated_score",
        provenance=provenance,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data_perp",
        help="Market data/artifact root.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()
    path = build_reference(
        data_root=Path(args.data_root),
        run_id=str(args.run_id),
        market_mode=str(args.market_mode),
    )
    print(path)


if __name__ == "__main__":
    main()
