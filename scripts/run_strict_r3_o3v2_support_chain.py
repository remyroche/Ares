#!/usr/bin/env python3
"""Run the bounded O3-v2 support-label funnel sequentially and immutably.

This is an offline research orchestrator.  It waits for the already-running
Stage-2 screen, selects on its declared Q4 development block, then runs the
policy-state and combined-weight screens.  A final frozen support contract is
scored forward only after the development choice is sealed.  It never imports
or mutates live inference, MC1, admission, portfolio, or canonical artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_o3v2_support_funnel as impl  # noqa: E402
import run_strict_r3_o3v2_support_funnel_v3 as v3  # noqa: E402
import select_strict_r3_o3v2_support as selector  # noqa: E402


FEATURE_ROOT = Path("data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/target_free_monthly")
SEMANTIC_ROOT = Path("data_perp/artifacts/strict_r3_o3v2_semantics_20260824_v1")
POLICY_PATH = Path("data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet")
BUNDLE_ROOT = Path("data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_reconstruction_2025_2026_20260816_v4")
TARGET_METRICS = Path("data_perp/artifacts/strict_r3_o3v2_target_funnel_20260824_v3_exact_full_detached/target_funnel_metrics.parquet")
TARGET_SELECTION = Path("data_perp/artifacts/strict_r3_o3v2_target_selection_20260824_v3/selected_target_contracts.json")
STAGE2 = Path("data_perp/artifacts/strict_r3_o3v2_support_stage2_20260824_v3_exact")
STAGE3 = Path("data_perp/artifacts/strict_r3_o3v2_support_stage3_20260824_v3_exact")
BUNDLES = Path("data_perp/artifacts/strict_r3_o3v2_support_bundles_20260824_v3_exact")
SELECTION_STAGE2 = Path("data_perp/artifacts/strict_r3_o3v2_support_selection_stage2_20260824_v3")
SELECTION_FINAL = Path("data_perp/artifacts/strict_r3_o3v2_support_selection_final_20260824_v3")
FORWARD = Path("data_perp/artifacts/strict_r3_o3v2_support_forward_20260824_v3_exact")

DEVELOPMENT = tuple(f"2025-{month:02d}" for month in (10, 11, 12))
FORWARD_MONTHS = tuple(f"2026-{month:02d}" for month in range(1, 8))
STAGE3_ARMS = (
    "S0_uniform", "S5_tbm_coarse", "S5_exit4_policy", "S5_exit5_policy", "S5_sequential_policy",
)
BUNDLE_ARMS = (
    "S0_uniform", "SB1_error_archetype", "SB2_error_policy_state",
    "SB3_error_semantic", "SB3_error_pair_semantic",
)


def _configure_v3() -> None:
    impl.SCHEMA = v3.SCHEMA
    impl.SUPPORT_ARMS = v3.SUPPORT_ARMS
    impl._components = v3._components
    impl._weights = v3._weights


def _wait_manifest(root: Path) -> None:
    while not (root / "run_manifest.json").exists():
        if not root.exists():
            raise FileNotFoundError(root)
        print(json.dumps({"event": "waiting", "root": str(root)}), flush=True)
        time.sleep(30)


def _run_support(*, out: Path, target_arms: tuple[str, ...], support_arms: tuple[str, ...], months: tuple[str, ...], pairs: tuple[tuple[str, str], ...] | None = None) -> None:
    if (out / "run_manifest.json").exists():
        print(json.dumps({"event": "already_finalized", "root": str(out)}), flush=True)
        return
    _configure_v3()
    impl.run(
        feature_root=FEATURE_ROOT, semantic_root=SEMANTIC_ROOT, policy_path=POLICY_PATH,
        bundle_root=BUNDLE_ROOT, out=out,
        months=tuple(pd.Timestamp(f"{month}-01", tz="UTC") for month in months),
        target_arms=target_arms, support_arms=support_arms, pairs=pairs,
        query_mode="exact_timestamp_side", resume=out.exists(),
    )


def _select(*, inputs: tuple[Path, ...], out: Path) -> None:
    if (out / "selected_support_contracts.json").exists():
        return
    selector.run(
        target_metrics=TARGET_METRICS,
        support_metrics=tuple(source / "support_funnel_metrics.parquet" for source in inputs),
        out=out, months=DEVELOPMENT,
    )


def main() -> None:
    target_selection = json.loads(TARGET_SELECTION.read_text())
    target_arms = tuple(target_selection["selected"])
    if len(target_arms) != 2:
        raise AssertionError(f"expected exactly two distinct selected target concepts, got {target_arms}")
    _wait_manifest(STAGE2)
    _select(inputs=(STAGE2,), out=SELECTION_STAGE2)
    _run_support(out=STAGE3, target_arms=target_arms, support_arms=STAGE3_ARMS, months=DEVELOPMENT)
    _run_support(out=BUNDLES, target_arms=target_arms, support_arms=BUNDLE_ARMS, months=DEVELOPMENT)
    _select(inputs=(STAGE2, STAGE3, BUNDLES), out=SELECTION_FINAL)
    final = json.loads((SELECTION_FINAL / "selected_support_contracts.json").read_text())
    pairs = tuple((str(row["target_arm"]), str(row["support_arm"])) for row in final["selected"])
    _run_support(
        out=FORWARD, target_arms=tuple(dict.fromkeys(pair[0] for pair in pairs)),
        support_arms=tuple(dict.fromkeys(pair[1] for pair in pairs)), months=FORWARD_MONTHS, pairs=pairs,
    )
    (FORWARD / "support_chain_receipt.json").write_text(json.dumps({
        "scope": "offline O3-v2 support funnel only; no MC1/live/canonical mutation",
        "development_months": DEVELOPMENT, "forward_months": FORWARD_MONTHS,
        "final_selected_pairs": [{"target_arm": target_arm, "support_arm": support_arm} for target_arm, support_arm in pairs],
        "selection": "development-only final support choice; forward run is diagnostic only",
    }, indent=2, sort_keys=True))
    print(json.dumps({"event": "support_chain_finalized", "pairs": pairs}), flush=True)


if __name__ == "__main__":
    main()
