#!/usr/bin/env python3
"""Seal v170: post-fill rollback and direct Kraken inventory truth."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/seal_strict_r3_v166_direct_predecessor_fail_closed.py"
spec = importlib.util.spec_from_file_location("seal_v166", BASE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.OLD_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v144_bcf_current_dual_"
    "samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_"
    "execution_parallel_source_io_lockstep_failclosed_bounded_parquet_io_"
    "book_snapshot_vwap.json"
)
module.NEW_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_v145_v170_postfill_directpos.json"
)
module.OLD_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v145_v169_book_snapshot_vwap_"
    "live.json"
)
module.NEW_EXECUTION = Path(
    "config/strict_r3_kraken_live_execution_v146_v170_postfill_directpos_live.json"
)
module.OLD_AUTH = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260823_v71_v144_"
    "book_snapshot_vwap_live.json"
)
module.NEW_AUTH = Path(
    "config/strict_r3_kraken_live_activation_authorization_20260823_v72_v145_"
    "postfill_directpos_live.json"
)
module.OLD_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v102_v169_book_snapshot_vwap_"
    "live.json"
)
module.NEW_STATE = Path(
    "data_perp/live/strict_r3_kraken_live_state_v103_v170_postfill_directpos_live.json"
)
module.RECEIPT = Path(
    "data_perp/artifacts/strict_r3_postfill_rollback_direct_positions_"
    "reseal_20260823_v1/run_manifest.json"
)
module.CHANGED = ["extreme_price_movements/inference/strict_r3_live_execution.py"]
module.VERSION_LABEL = "v170 post-fill rollback and direct Kraken positions"
module.RESEAL_REASON = (
    "A filled entry is now flattened if native protection cannot be built, and "
    "Kraken inventory reconciliation uses the exchange's direct openpositions "
    "endpoint rather than stale generic-adapter rows. Models, calibration, "
    "admission, auction, sizing, and exit geometry are unchanged."
)


if __name__ == "__main__":
    module.main()
