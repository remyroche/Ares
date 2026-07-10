#!/usr/bin/env python3
"""Score unsupervised regime features for side x archetype EV calibration.

This report is intentionally local: feature/composite relevance and optional
LGBM relevance models are trained/evaluated per ``side_name x archetype``.
The top-k thresholds are global so the denominator matches what policy replay
and inference trade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.evm_latent_state_discovery import select_evm_state_feature_columns  # noqa: E402
from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (  # noqa: E402
    EconomicRegimeRelevanceConfig,
    materialize_composite_features,
    run_economic_regime_relevance,
)
from scripts.report_evm_latent_state_archetype_diagnostics import (  # noqa: E402
    DEFAULT_HANDOFF,
    DEFAULT_META_RUN,
    _prepare_panel,
)


DEFAULT_OUT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_"
    "ae3000_nocrossfit_k34567_payload300k_20260706/"
    "unsupervised_regime_economic_relevance_20260710"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-run", type=Path, default=DEFAULT_META_RUN)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all-months", nargs="+", default=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--include-aegmm", action="store_true")
    parser.add_argument("--max-feature-cols", type=int, default=160)
    parser.add_argument("--min-group-rows", type=int, default=300)
    parser.add_argument("--min-population-rows", type=int, default=60)
    parser.add_argument("--min-state-rows", type=int, default=20)
    parser.add_argument("--max-features-per-group", type=int, default=80)
    parser.add_argument("--max-features-for-composites", type=int, default=10)
    parser.add_argument("--max-composites-per-group-task", type=int, default=80)
    parser.add_argument("--min-candidate-score", type=float, default=0.03)
    parser.add_argument("--trade-top-fraction", type=float, default=0.10)
    parser.add_argument("--promote-outer-top-fraction", type=float, default=0.20)
    parser.add_argument("--diagnostic-promote-outer-top-fraction", type=float, default=0.30)
    parser.add_argument("--temporal-score-weight", type=float, default=0.25)
    parser.add_argument("--disable-lgbm", action="store_true")
    parser.add_argument("--lgbm-min-rows", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, _loaded_features = _prepare_panel(
        meta_run=args.meta_run,
        handoff=args.handoff,
        months=list(args.all_months),
        include_aegmm=bool(args.include_aegmm),
        max_feature_cols=int(args.max_feature_cols),
    )
    required = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "month",
        "week_start",
        "score_meta_base_soft_label",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    if "stop_or_adverse" not in panel.columns:
        if "full_stop_loss" in panel.columns:
            panel["stop_or_adverse"] = pd.to_numeric(panel["full_stop_loss"], errors="coerce").fillna(0.0).astype("int8")
        else:
            panel["stop_or_adverse"] = 0
    feature_cols = select_evm_state_feature_columns(
        panel,
        include_aegmm=bool(args.include_aegmm),
        required_columns=required,
        max_columns=int(args.max_feature_cols),
    )
    config = EconomicRegimeRelevanceConfig(
        min_group_rows=int(args.min_group_rows),
        min_population_rows=int(args.min_population_rows),
        min_state_rows=int(args.min_state_rows),
        max_features_per_group=int(args.max_features_per_group),
        max_features_for_composites=int(args.max_features_for_composites),
        max_composites_per_group_task=int(args.max_composites_per_group_task),
        min_candidate_score=float(args.min_candidate_score),
        trade_top_fraction=float(args.trade_top_fraction),
        negative_top_fraction=float(args.trade_top_fraction),
        positive_outer_top_fraction=float(args.promote_outer_top_fraction),
        positive_diagnostic_outer_fraction=float(args.diagnostic_promote_outer_top_fraction),
        temporal_score_weight=float(args.temporal_score_weight),
        lgbm_enabled=not bool(args.disable_lgbm),
        lgbm_min_rows=int(args.lgbm_min_rows),
    )
    result = run_economic_regime_relevance(panel, feature_cols, config=config)
    result.feature_metrics.to_csv(args.output_dir / "feature_relevance_by_side_archetype.csv", index=False)
    result.composite_metrics.to_csv(args.output_dir / "composite_relevance_by_side_archetype.csv", index=False)
    result.lgbm_feature_metrics.to_csv(args.output_dir / "local_lgbm_feature_relevance.csv", index=False)
    result.lgbm_model_metrics.to_csv(args.output_dir / "local_lgbm_model_metrics.csv", index=False)
    result.selected_candidates.to_csv(args.output_dir / "ebm_selected_candidate_features.csv", index=False)
    composite_frame = materialize_composite_features(panel, result.composite_definitions)
    composite_out = panel[["__ts__", "__symbol__", "side_name"]].reset_index(drop=True).copy()
    if not composite_frame.empty:
        composite_out = pd.concat([composite_out, composite_frame.reset_index(drop=True)], axis=1)
    composite_out.to_parquet(args.output_dir / "materialized_economic_composite_features.parquet", index=False)
    (args.output_dir / "ebm_candidate_feature_manifest.json").write_text(
        json.dumps(result.ebm_candidate_manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    manifest = {
        "generated_by": "report_unsupervised_regime_economic_relevance.py",
        "meta_run": str(args.meta_run),
        "handoff": str(args.handoff),
        "all_months": list(args.all_months),
        "include_aegmm": bool(args.include_aegmm),
        "feature_count": int(len(feature_cols)),
        "feature_metric_rows": int(len(result.feature_metrics)),
        "composite_metric_rows": int(len(result.composite_metrics)),
        "composite_definitions": int(len(result.composite_definitions)),
        "local_lgbm_feature_rows": int(len(result.lgbm_feature_metrics)),
        "local_lgbm_model_rows": int(len(result.lgbm_model_metrics)),
        "selected_candidate_rows": int(len(result.selected_candidates)),
        "outputs": {
            "feature_relevance": str(args.output_dir / "feature_relevance_by_side_archetype.csv"),
            "composite_relevance": str(args.output_dir / "composite_relevance_by_side_archetype.csv"),
            "local_lgbm_feature_relevance": str(args.output_dir / "local_lgbm_feature_relevance.csv"),
            "local_lgbm_model_metrics": str(args.output_dir / "local_lgbm_model_metrics.csv"),
            "ebm_candidate_manifest": str(args.output_dir / "ebm_candidate_feature_manifest.json"),
            "materialized_composites": str(args.output_dir / "materialized_economic_composite_features.parquet"),
        },
        "topk_contract": {
            "demote": "global top10% by score; target is bad/dirty/non-clean/negative EV top10",
            "promote": "global top20% excluding top10%; target is clean positive EV near-threshold",
            "diagnostics": "global top15/top20 demotion and top30-not-top10 promotion are reported but not selected by default",
            "locality": "all feature/composite relevance and LGBM models are per side x archetype",
            "temporal_alignment": "feature states are additionally scored by best/worst day/week alignment and aligned streaks per side x archetype",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({"event": "unsupervised_regime_economic_relevance_done", **manifest}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
