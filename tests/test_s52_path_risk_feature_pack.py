from __future__ import annotations

import pandas as pd

from scripts import build_s52_path_risk_feature_pack as pack


def test_path_risk_feature_pack_scores_and_selects(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = []
    for segment in ("long", "short", "all"):
        for i in range(12):
            rows.append(
                {
                    "segment": segment,
                    "feature": f"{segment}_feature_{i}",
                    "feature_family": "test",
                    "dominant_polarity": "positive",
                    "mean_top10_mean_first_touch_net": 0.003 - 0.0005 * i,
                    "mean_top10_ev_weighted_first_touch_precision": 0.7 - 0.01 * i,
                    "mean_top10_first_pass_good_rate": 0.65 - 0.01 * i,
                    "mean_top10_first_touch_full_path_bad_mae_1r_rate": 0.35 + 0.03 * i,
                    "mean_top10_mfe_1r_before_mae_1r_rate": 0.8 - 0.01 * i,
                    "mean_top10_mae_1r_before_mfe_1r_rate": 0.1 + 0.01 * i,
                    "mean_top10_timeout_rate": 0.02 + 0.005 * i,
                }
            )
    pd.DataFrame(rows).to_csv(source_dir / "s52_feature_learnability_feature_summary.csv", index=False)
    pd.DataFrame({"feature": ["previous_a", "previous_b"]}).to_csv(
        source_dir / "s52_learnability_ranker_feature_list_top360.csv",
        index=False,
    )

    output_dir = tmp_path / "out"
    result = pack.run(
        source_dir=source_dir,
        output_dir=output_dir,
        top_per_segment_objective=3,
        top_per_segment_path=3,
        top_per_segment_balanced=3,
        max_total_features=20,
    )

    selected = pd.read_csv(result["selected"])
    feature_list = pd.read_csv(result["feature_list"])
    assert not selected.empty
    assert "previous_a" in set(feature_list["feature"])
    assert selected["selection_reason"].str.contains("path_clean|opportunity|balanced|previous").all()
    assert (output_dir / "s52_path_risk_feature_pack.md").exists()
