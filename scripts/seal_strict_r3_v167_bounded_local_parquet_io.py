#!/usr/bin/env python3
"""Seal v167: bound local Parquet fan-out while preserving source semantics."""

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
    "config/strict_r3_inference_overlay_long_20260801_v141_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_parallel_source_io_lockstep_failclosed.json"
)
module.NEW_OVERLAY = Path(
    "config/strict_r3_inference_overlay_long_20260801_v142_bcf_current_dual_samebundle21d_direct_15m_reference_lockstep_geometry_feature_parity_warm_execution_parallel_source_io_lockstep_failclosed_bounded_parquet_io.json"
)
module.OLD_EXECUTION = Path("config/strict_r3_kraken_live_execution_v142_v166_lockstep_failclosed_live.json")
module.NEW_EXECUTION = Path("config/strict_r3_kraken_live_execution_v143_v167_bounded_parquet_io_live.json")
module.OLD_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v68_v141_lockstep_failclosed_live.json")
module.NEW_AUTH = Path("config/strict_r3_kraken_live_activation_authorization_20260822_v69_v142_bounded_parquet_io_live.json")
module.OLD_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v99_v166_lockstep_failclosed_live.json")
module.NEW_STATE = Path("data_perp/live/strict_r3_kraken_live_state_v100_v167_bounded_parquet_io_live.json")
module.RECEIPT = Path("data_perp/artifacts/strict_r3_bounded_parquet_io_reseal_20260822_v1/run_manifest.json")
module.CHANGED = [
    "scripts/materialize_strict_r3_target_free_hourly_grid_v2.py",
    "scripts/run_tp6_sl4_exact170_canonical_consensus.py",
]
module.VERSION_LABEL = "v167 bounded local-Parquet source I/O"
module.RESEAL_REASON = (
    "Bound candidate-grid and panel local-Parquet reader pools to four workers. "
    "This prevents metadata/I/O saturation of mixed-age shards while retaining "
    "the exact source precedence, candidate values, universe order, eligibility, "
    "models, calibration, admission, portfolio and exits."
)

if __name__ == "__main__":
    module.main()
