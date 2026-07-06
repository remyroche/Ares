from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_promoted_cross_asset_regularized_meta_ablation import _comparison_rows  # noqa: E402


def _manifest(selector: str, *, exec10: float, clean10: float, bad10: float, exec30: float, bad30: float, ap: float) -> dict:
    return {
        "best_selector": {
            "selector": selector,
            "meta_smoke_status": "candidate_for_deeper_meta_eval",
            "mean_keep010_exec_margin": exec10,
            "mean_keep010_clean_exec_precision": clean10,
            "mean_keep010_full_path_bad_mae": bad10,
            "mean_keep010_timeout": 0.01,
            "mean_keep010_oracle_recall": 0.20,
            "mean_keep030_exec_margin": exec30,
            "mean_keep030_clean_exec_precision": 0.60,
            "mean_keep030_full_path_bad_mae": bad30,
            "mean_keep030_timeout": 0.01,
            "mean_keep030_oracle_recall": 0.40,
            "mean_ap_clean_exec": ap,
            "mean_auc_clean_exec": 0.60,
        }
    }


def test_regularized_meta_comparison_requires_baseline_relative_improvement(tmp_path: Path) -> None:
    baseline = _manifest("base_score", exec10=0.010, clean10=0.62, bad10=0.56, exec30=0.007, bad30=0.55, ap=0.55)
    profiles = {
        "good": _manifest("meta_clean_exec", exec10=0.011, clean10=0.64, bad10=0.52, exec30=0.0071, bad30=0.555, ap=0.56),
        "bad": _manifest("meta_clean_exec", exec10=0.012, clean10=0.61, bad10=0.60, exec30=0.0072, bad30=0.59, ap=0.54),
    }

    comparison = _comparison_rows(profiles, baseline)

    by_profile = comparison.set_index("profile")
    assert by_profile.loc["good", "regularized_gate_status"] == "candidate_for_deeper_meta_eval"
    assert by_profile.loc["bad", "regularized_gate_status"] == "diagnostic_or_fail"
    assert by_profile.loc["good", "delta_vs_baseline__mean_keep010_exec_margin"] > 0.0
