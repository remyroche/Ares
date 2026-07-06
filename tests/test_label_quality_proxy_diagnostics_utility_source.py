from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _path_metrics


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__barrier_pct__": [0.01, 0.02],
            "__mfe_ret__": [0.02, 0.01],
            "__mae_ret__": [0.003, 0.004],
            "__bars_policy__": [4, 5],
            "__y_ret__": [0.012, -0.006],
            "__y_bin__": [1.0, 0.0],
            "__is_timeout__": [0.0, 1.0],
            "side": [1, -1],
        }
    )


def test_path_metrics_prefers_policy_net_utility_when_present() -> None:
    frame = _base_frame()
    frame["__u_policy_net__"] = [0.03, -0.02]

    metrics = _path_metrics(frame)

    assert metrics.attrs["utility_source"] == "__u_policy_net__"
    assert metrics["u_policy_net"].tolist() == [0.03, -0.02]


def test_path_metrics_falls_back_to_y_ret_for_utility() -> None:
    metrics = _path_metrics(_base_frame())

    assert metrics.attrs["utility_source"] == "__y_ret__"
    assert metrics["u_policy_net"].tolist() == [0.012, -0.006]
