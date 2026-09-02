#!/usr/bin/env python3
"""Seed compact fixed-FFD state from one sealed causal source panel.

This utility computes only the four fixed close transforms consumed by the
strict-R3 feature graph.  It does not construct the broader feature graph and
therefore provides an exact low-memory migration path for older state bundles.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.features import (  # noqa: E402
    _safe_log_df,
    _transform_close_fixed_ffd,
)
from scripts.update_strict_r3_feature_panel_state import STATE_SCHEMA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    state = joblib.load(args.panel_state)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported source-panel state")
    close = state["panel"]["close"].astype(np.float32)
    cfg = dict(CFG)
    cfg.update(
        {
            "live_fixed_ffd_state_enabled": True,
            "live_fixed_ffd_state_dir": str(args.cache_dir / "fixed_ffd"),
            "feature_fixed_ffd_output_history_rows": 768,
        }
    )
    thres = float(cfg.get("ffd_thres", 1e-5))
    _transform_close_fixed_ffd(
        close,
        d=float(cfg.get("ffd_d_base", 0.4)),
        _label="close",
        already_logged=False,
        thres=thres,
        cfg=cfg,
    )
    logged = _safe_log_df(close)
    for d in sorted(set(float(value) for value in cfg.get("ffd_d_values", [0.4, 0.5, 0.6]))):
        tag = f"{int(round(d * 10)):02d}"
        _transform_close_fixed_ffd(
            logged,
            d=d,
            _label=f"close_d{tag}",
            already_logged=True,
            thres=thres,
            cfg=cfg,
        )
        gc.collect()
    files = sorted((args.cache_dir / "fixed_ffd").glob("fixed_ffd_state.*.npz"))
    if len(files) != 4:
        raise AssertionError(f"expected four fixed-FFD states, found {len(files)}")
    receipt = {
        "schema": "strict_r3_fixed_ffd_state_bootstrap_v1",
        "panel_state": str(args.panel_state),
        "last_timestamp": close.index[-1].isoformat(),
        "symbols": int(close.shape[1]),
        "history_rows": int(close.shape[0]),
        "state_files": [str(path) for path in files],
    }
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
