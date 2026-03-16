from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.trigger_discovery import (
    TriggerDiscoveryConfig,
    current_trigger_feature_inventory,
    current_trigger_template_inventory,
)
DOC_PATH = REPO_ROOT / "docs" / "feature_trigger_gap_review.md"
JSON_PATH = REPO_ROOT / "artifacts" / "feature_trigger_gap_review.json"
FEATURE_CSV_PATH = REPO_ROOT / "artifacts" / "feature_trigger_inventory.csv"
TRIGGER_CSV_PATH = REPO_ROOT / "artifacts" / "trigger_template_inventory.csv"


TARGET_FEATURES: List[Dict[str, Any]] = [
    {"target_name": "range", "family": "volatility_range"},
    {"target_name": "true_range", "family": "volatility_range"},
    {"target_name": "atr_14", "family": "volatility_range"},
    {"target_name": "atr_100", "family": "volatility_range"},
    {"target_name": "range_atr", "family": "volatility_range"},
    {"target_name": "compression_ratio", "family": "volatility_range"},
    {"target_name": "rolling_range_5", "family": "volatility_range"},
    {"target_name": "rolling_range_10", "family": "volatility_range"},
    {"target_name": "rolling_range_20", "family": "volatility_range"},
    {"target_name": "body", "family": "candle_geometry"},
    {"target_name": "body_ratio", "family": "candle_geometry"},
    {"target_name": "upper_wick", "family": "candle_geometry"},
    {"target_name": "lower_wick", "family": "candle_geometry"},
    {"target_name": "upper_wick_ratio", "family": "candle_geometry"},
    {"target_name": "lower_wick_ratio", "family": "candle_geometry"},
    {"target_name": "close_location_in_bar", "family": "candle_geometry"},
    {"target_name": "open_location_in_bar", "family": "candle_geometry", "optional": True},
    {"target_name": "signed_body_ratio", "family": "candle_geometry", "optional": True},
    {"target_name": "ema_10", "family": "trend_distance"},
    {"target_name": "ema_20", "family": "trend_distance"},
    {"target_name": "ema_30", "family": "trend_distance"},
    {"target_name": "ema_50", "family": "trend_distance"},
    {"target_name": "ema_slope_ema20_3", "family": "trend_distance"},
    {"target_name": "ema_slope_ema20_5", "family": "trend_distance"},
    {"target_name": "ema_slope_ema50_3", "family": "trend_distance", "optional": True},
    {"target_name": "distance_to_ema10", "family": "trend_distance"},
    {"target_name": "distance_to_ema20", "family": "trend_distance"},
    {"target_name": "distance_to_ema30", "family": "trend_distance"},
    {"target_name": "distance_to_ema20_atr", "family": "trend_distance"},
    {"target_name": "distance_to_ema50_atr", "family": "trend_distance"},
    {"target_name": "trend_alignment_ema20_gt_ema50", "family": "trend_distance"},
    {"target_name": "returns_1", "family": "momentum"},
    {"target_name": "returns_3", "family": "momentum"},
    {"target_name": "returns_5", "family": "momentum"},
    {"target_name": "returns_10", "family": "momentum"},
    {"target_name": "acceleration_close", "family": "momentum"},
    {"target_name": "acceleration_close_atr", "family": "momentum"},
    {"target_name": "momentum_sign_N", "family": "momentum", "optional": True},
    {"target_name": "macd_histogram", "family": "momentum", "optional": True},
    {"target_name": "rsi_14", "family": "momentum", "optional": True},
    {"target_name": "volume_ma_20", "family": "volume"},
    {"target_name": "volume_spike", "family": "volume"},
    {"target_name": "volume_zscore_rolling", "family": "volume", "optional": True},
    {"target_name": "rolling_high_5", "family": "structure"},
    {"target_name": "rolling_high_10", "family": "structure"},
    {"target_name": "rolling_high_20", "family": "structure"},
    {"target_name": "rolling_low_5", "family": "structure"},
    {"target_name": "rolling_low_10", "family": "structure"},
    {"target_name": "rolling_low_20", "family": "structure"},
    {"target_name": "close_gt_rolling_high_5", "family": "structure", "optional": True},
    {"target_name": "close_lt_rolling_low_5", "family": "structure", "optional": True},
    {"target_name": "high_gt_rolling_high_5", "family": "structure"},
    {"target_name": "low_lt_rolling_low_5", "family": "structure"},
    {"target_name": "bullish_bar", "family": "bar_state"},
    {"target_name": "bearish_bar", "family": "bar_state"},
    {"target_name": "prior_bullish_bar", "family": "bar_state"},
    {"target_name": "prior_bearish_bar", "family": "bar_state"},
    {"target_name": "inside_bar", "family": "bar_state", "optional": True},
    {"target_name": "outside_bar", "family": "bar_state", "optional": True},
]


TARGET_TRIGGERS: List[Dict[str, Any]] = [
    {"target_name": "close_crosses_above_ema", "family": "pullback_recovery"},
    {"target_name": "ema_reclaim_touch", "family": "pullback_recovery"},
    {"target_name": "reclaim_after_opposite_bar", "family": "pullback_recovery"},
    {"target_name": "close_in_extreme_of_range", "family": "pullback_recovery"},
    {"target_name": "simple_close_breakout", "family": "breakout"},
    {"target_name": "close_gt_rolling_extreme", "family": "breakout"},
    {"target_name": "high_break_close_near_extreme", "family": "breakout"},
    {"target_name": "expansion_body_breakout", "family": "breakout"},
    {"target_name": "expansion_bar", "family": "expansion_impulse"},
    {"target_name": "impulse_bar", "family": "expansion_impulse"},
    {"target_name": "sweep_reversal", "family": "sweep_reversal"},
    {"target_name": "relaxed_sweep", "family": "sweep_reversal"},
    {"target_name": "compression_release", "family": "compression_release"},
    {"target_name": "compressed_breakout_up_down", "family": "compression_release"},
]


FEATURE_ALIAS_MAP: Dict[str, List[str]] = {
    "range_atr": ["intrabar_range_atr"],
    "distance_to_ema20_atr": ["distance_from_ema_atr"],
    "returns_1": ["ret_1"],
    "compression_ratio": ["atr_compression_ratio", "vol_compression_ratio"],
    "acceleration_close": ["acceleration", "acceleration_of_move"],
    "acceleration_close_atr": ["acceleration_norm"],
}


def _load_current_feature_inventory() -> List[Dict[str, Any]]:
    current = list(current_trigger_feature_inventory())
    current.extend(
        [
            {
                "feature_name": "hl_range",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "rolling robust z-score of high-low range",
                "parameterization": None,
                "dimensionless": False,
                "normalized": True,
                "notes": "approximate range proxy for regime search",
            },
            {
                "feature_name": "intrabar_range_atr",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "intrabar range normalized by approximate ATR",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
                "notes": "approximate range_atr proxy",
            },
            {
                "feature_name": "compression_expansion_transition",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "range spike divided by rolling bollinger width proxy",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
            },
            {
                "feature_name": "distance_from_ema_atr",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "distance from SMA/EMA proxy normalized by ATR proxy",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
            },
            {
                "feature_name": "volume_robust_z",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "rolling robust z-score of volume",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
            },
            {
                "feature_name": "breakout_distance_up_atr",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "distance from shifted trailing high normalized by ATR proxy",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
            },
            {
                "feature_name": "breakout_distance_down_atr",
                "source_file": "extreme_price_movements/mask_optimiser.py",
                "source_function": "_compute_z_cache",
                "formula_or_description": "distance from shifted trailing low normalized by ATR proxy",
                "parameterization": None,
                "dimensionless": True,
                "normalized": True,
            },
            {
                "feature_name": "true_range_percentile",
                "source_file": "extreme_price_movements/ridge_regime_event_assessment.py",
                "source_function": "build_regime_features",
                "formula_or_description": "rolling percentile rank of true range",
                "parameterization": 168,
                "dimensionless": True,
                "normalized": True,
            },
        ]
    )
    for row in current:
        row.setdefault("source_file", "extreme_price_movements/trigger_discovery.py")
        row.setdefault("notes", "")
    return current


def _load_current_trigger_inventory() -> List[Dict[str, Any]]:
    current = current_trigger_template_inventory(
        TriggerDiscoveryConfig(enable_compression_release_triggers=True)
    )
    for row in current:
        row.setdefault("source_file", "extreme_price_movements/trigger_discovery.py")
        row.setdefault("notes", "")
    return current


def _compare_features(
    current_inventory: List[Dict[str, Any]],
    target_inventory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    current_names = {row["feature_name"]: row for row in current_inventory}
    out: List[Dict[str, Any]] = []
    for target in target_inventory:
        name = target["target_name"]
        matched_name: Optional[str] = None
        status = "missing"
        notes = ""
        if name in current_names:
            matched_name = name
            status = "exact_match"
        else:
            for alias in FEATURE_ALIAS_MAP.get(name, []):
                if alias in current_names:
                    matched_name = alias
                    status = "approximate_match"
                    notes = f"Mapped to approximate equivalent `{alias}`."
                    break
        if status == "missing" and target.get("optional", False):
            notes = "Optional target not implemented."
        out.append(
            {
                "target_name": name,
                "status": status,
                "matched_current_name": matched_name,
                "notes": notes,
            }
        )
    return out


def _compare_triggers(
    current_inventory: List[Dict[str, Any]],
    target_inventory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    current_names = {row["trigger_name"]: row for row in current_inventory}
    out: List[Dict[str, Any]] = []
    for target in target_inventory:
        name = target["target_name"]
        status = "exact_match" if name in current_names else "missing"
        out.append(
            {
                "target_name": name,
                "status": status,
                "matched_current_name": name if name in current_names else None,
                "notes": "" if status == "exact_match" else "No trigger template found.",
            }
        )
    return out


def build_gap_review_payload() -> Dict[str, Any]:
    current_features = _load_current_feature_inventory()
    current_triggers = _load_current_trigger_inventory()
    feature_comparison = _compare_features(current_features, TARGET_FEATURES)
    trigger_comparison = _compare_triggers(current_triggers, TARGET_TRIGGERS)
    added_features = [
        row["target_name"]
        for row in feature_comparison
        if row["status"] == "exact_match"
        and row["target_name"]
        in {
            "range",
            "atr_14",
            "atr_100",
            "rolling_range_5",
            "rolling_range_10",
            "rolling_range_20",
            "body",
            "open_location_in_bar",
            "signed_body_ratio",
            "ema_50",
            "ema_slope_ema20_3",
            "ema_slope_ema20_5",
            "distance_to_ema10",
            "distance_to_ema20",
            "distance_to_ema30",
            "distance_to_ema20_atr",
            "distance_to_ema50_atr",
            "trend_alignment_ema20_gt_ema50",
            "returns_1",
            "returns_3",
            "returns_5",
            "returns_10",
            "acceleration_close",
            "acceleration_close_atr",
            "volume_ma_20",
            "bullish_bar",
            "bearish_bar",
            "prior_bullish_bar",
            "prior_bearish_bar",
            "inside_bar",
            "outside_bar",
        }
    ]
    added_triggers = [
        row["target_name"]
        for row in trigger_comparison
        if row["target_name"]
        in {
            "ema_reclaim_touch",
            "simple_close_breakout",
            "expansion_bar",
            "impulse_bar",
            "relaxed_sweep",
            "compression_release",
            "compressed_breakout_up_down",
        }
    ]
    return {
        "features": {
            "current": current_features,
            "target": TARGET_FEATURES,
            "comparison": feature_comparison,
        },
        "triggers": {
            "current": current_triggers,
            "target": TARGET_TRIGGERS,
            "comparison": trigger_comparison,
        },
        "added": {
            "features": added_features,
            "triggers": added_triggers,
        },
    }


def _markdown_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        data_lines.append("| " + " | ".join(vals) + " |")
    return "\n".join([header_line, sep_line, *data_lines])


def render_markdown_report(payload: Dict[str, Any]) -> str:
    feature_cmp = payload["features"]["comparison"]
    trigger_cmp = payload["triggers"]["comparison"]
    missing_features = [row["target_name"] for row in feature_cmp if row["status"] == "missing"]
    missing_triggers = [row["target_name"] for row in trigger_cmp if row["status"] == "missing"]
    lines = [
        "# Feature/Trigger Gap Review",
        "",
        "## Summary",
        "",
        f"- Current feature inventory items: {len(payload['features']['current'])}",
        f"- Current trigger inventory items: {len(payload['triggers']['current'])}",
        f"- Target feature items: {len(payload['features']['target'])}",
        f"- Target trigger items: {len(payload['triggers']['target'])}",
        f"- Missing feature items: {len(missing_features)}",
        f"- Missing trigger items: {len(missing_triggers)}",
        "",
        "## Current Feature Inventory",
        "",
        _markdown_table(
            payload["features"]["current"],
            ["feature_name", "source_file", "source_function", "parameterization", "formula_or_description"],
        ),
        "",
        "## Current Trigger Inventory",
        "",
        _markdown_table(
            payload["triggers"]["current"],
            ["trigger_family", "trigger_name", "source_file", "source_function", "params", "semantic_description"],
        ),
        "",
        "## Target Feature List",
        "",
        _markdown_table(payload["features"]["target"], ["target_name", "family", "optional"]),
        "",
        "## Target Trigger List",
        "",
        _markdown_table(payload["triggers"]["target"], ["target_name", "family"]),
        "",
        "## Match Table For Features",
        "",
        _markdown_table(feature_cmp, ["target_name", "status", "matched_current_name", "notes"]),
        "",
        "## Match Table For Triggers",
        "",
        _markdown_table(trigger_cmp, ["target_name", "status", "matched_current_name", "notes"]),
        "",
        "## Missing Items",
        "",
        f"- Features: {missing_features if missing_features else 'None'}",
        f"- Triggers: {missing_triggers if missing_triggers else 'None'}",
        "",
        "## Implementation Plan",
        "",
        "- Inventory current feature/trigger sources from the trigger discovery and regime search stack.",
        "- Compare against the target reference lists using exact and approximate matching rules.",
        "- Extend the canonical trigger feature frame with missing OHLCV primitives.",
        "- Add missing primitive trigger templates with config-driven toggles and long/short symmetry.",
        "- Regenerate the review artifacts after implementation.",
        "",
        "## Post-Implementation Status",
        "",
        f"- Added features: {payload['added']['features']}",
        f"- Added triggers: {payload['added']['triggers']}",
        f"- Remaining missing features: {missing_features if missing_features else 'None'}",
        f"- Remaining missing triggers: {missing_triggers if missing_triggers else 'None'}",
        "",
    ]
    return "\n".join(lines)


def write_review_artifacts() -> Dict[str, Any]:
    payload = build_gap_review_payload()
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(render_markdown_report(payload), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(payload["features"]["current"]).to_csv(FEATURE_CSV_PATH, index=False)
    pd.DataFrame(payload["triggers"]["current"]).to_csv(TRIGGER_CSV_PATH, index=False)
    return payload


def main() -> None:
    write_review_artifacts()


if __name__ == "__main__":
    main()
