from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_quality_label_walkforward_ablation import _load_ablation_frame  # noqa: E402


def _joined_subset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=2, freq="h", tz="UTC"),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "candidate_id": ["c0", "c1"],
            "side": [1, 1],
            "quality_label_v0": [1, 0],
            "sample_weight_base_v0": [1.0, 1.0],
            "__barrier_pct__": [0.01, 0.02],
            "__mfe_ret__": [0.03, 0.01],
            "__mae_ret__": [-0.003, -0.025],
            "__bars_to_mfe__": [2, 5],
            "__bars_policy__": [4, 24],
            "__y_ret__": [0.02, -0.01],
            "__y_bin__": [1, 0],
            "__is_timeout__": [0, 1],
            "__u_policy_net__": [0.018, -0.012],
        }
    )


def test_load_ablation_frame_accepts_prejoined_subset(tmp_path: Path) -> None:
    joined_path = tmp_path / "source_quality_clean_joined_subset.parquet"
    _joined_subset_frame().to_parquet(joined_path, index=False)
    (tmp_path / "manifest.json").write_text(
        """
{
  "subset_status": "pass",
  "overall_status": "warning",
  "warnings": ["broad_quality_coverage_low"],
  "join_report": {
    "quality_rows": 100,
    "label_rows": 2,
    "joined_rows": 2,
    "join_key": ["__ts__", "__symbol__"],
    "join_match_rate_vs_quality": 0.02,
    "join_match_rate_vs_labels": 1.0,
    "side_join_used": true
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    frame, join_report, input_report = _load_ablation_frame(
        quality_labels_path=tmp_path / "unused_quality.parquet",
        labels_path=tmp_path / "unused_labels",
        joined_subset_path=joined_path,
    )

    assert len(frame) == 2
    assert input_report["mode"] == "prejoined_clean_subset"
    assert input_report["joined_subset_status"] == "pass"
    assert join_report["join_mode"] == "prejoined_subset"
    assert join_report["join_match_rate_vs_labels"] == 1.0
    assert join_report["join_match_rate_vs_quality"] == 0.02
    assert join_report["duplicate_candidate_id_rows"] == 0


def test_load_ablation_frame_rejects_non_joined_subset(tmp_path: Path) -> None:
    joined_path = tmp_path / "bad_subset.parquet"
    frame = _joined_subset_frame().drop(columns=["__u_policy_net__"])
    frame.to_parquet(joined_path, index=False)

    with pytest.raises(ValueError, match="missing outcome columns"):
        _load_ablation_frame(
            quality_labels_path=tmp_path / "unused_quality.parquet",
            labels_path=tmp_path / "unused_labels",
            joined_subset_path=joined_path,
        )
