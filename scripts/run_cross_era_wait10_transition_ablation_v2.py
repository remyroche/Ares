#!/usr/bin/env python3
"""Run the preregistered expanded cross-era Wait10 transition ablation.

v1 showed that 2025 and historical rows share high transition entropy and low
persistence while their Wait10 economics have opposite signs.  This wrapper
adds only the causal fields that distinguish those two state subtypes, then
uses the unchanged v1 fitting/evaluation engine and frozen book.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_cross_era_wait10_transition_ablation as engine

OUT = (
    ROOT
    / "data_perp/artifacts/cross_era_wait10_transition_ablation_20260730_v2"
)
SCHEMA = "cross_era_wait10_transition_ablation_v2"
SUBTYPE_FEATURES = (
    "btc_resilience_alt_weakness_gap",
    "downside_breadth_intensity",
    "breadth_dispersion",
    "short_default_damage_max_5d",
    "compression_quality_consistency",
    "short_default_damage_ema_5d",
    "btc_oi_dominance_z_ratio",
    "mkt_regime_change__funding__cumulative_change_2d",
    "fragmented_new_low_breadth",
    "state_context__state_age_hours",
)
EXPANDED_TRANSITIONS = (*engine.TRANSITION_COMMON, *SUBTYPE_FEATURES)
FEATURE_SETS = {
    "transition_common_v1": engine.TRANSITION_COMMON,
    "transition_subtype_only": SUBTYPE_FEATURES,
    "transition_expanded": EXPANDED_TRANSITIONS,
    "score_plus_transition_expanded": (
        *engine.SCORE_COMMON,
        *EXPANDED_TRANSITIONS,
    ),
}


def run(
    historical_root: Path = engine.HISTORICAL_ROOT,
    current_root: Path = engine.CURRENT_ROOT,
    handoff_root: Path = engine.HANDOFF_ROOT,
    calendar_root: Path = engine.CALENDAR_ROOT,
    output: Path = OUT,
) -> dict:
    engine.SCHEMA = SCHEMA
    engine.TRANSITION_COMMON = EXPANDED_TRANSITIONS
    engine.FEATURE_SETS = FEATURE_SETS
    manifest = engine.run(
        historical_root,
        current_root,
        handoff_root,
        calendar_root,
        output,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = (
        "SEALED_CROSS_ERA_EXPANDED_TRANSITION_DIAGNOSTIC_NO_PROMOTION"
    )
    manifest["contract"]["expanded_transition_features"] = (
        "preregistered after v1 state-cell diagnosis: BTC-alt resilience, "
        "breadth dispersion/intensity, compression quality, recent damage, "
        "funding structure and state age; all from the same sealed causal calendar"
    )
    manifest["feature_sets"] = {
        key: list(value) for key, value in FEATURE_SETS.items()
    }
    manifest["runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": engine.sha256(Path(__file__).resolve()),
        "shared_engine_path": str(Path(engine.__file__).resolve()),
        "shared_engine_sha256": engine.sha256(Path(engine.__file__).resolve()),
    }
    engine.write_json(manifest_path, manifest)
    (output / "manifest.sha256").write_text(
        f"{engine.sha256(manifest_path)}  manifest.json\n"
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--historical-root", type=Path, default=engine.HISTORICAL_ROOT)
    result.add_argument("--current-root", type=Path, default=engine.CURRENT_ROOT)
    result.add_argument("--handoff-root", type=Path, default=engine.HANDOFF_ROOT)
    result.add_argument("--calendar-root", type=Path, default=engine.CALENDAR_ROOT)
    result.add_argument("--output", type=Path, default=OUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            engine.safe(
                run(
                    args.historical_root,
                    args.current_root,
                    args.handoff_root,
                    args.calendar_root,
                    args.output,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
