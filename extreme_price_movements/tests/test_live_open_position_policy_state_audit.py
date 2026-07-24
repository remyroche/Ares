import json
from pathlib import Path

from scripts.audit_live_open_position_policy_state import _latest_payload, _close


def test_latest_payload_uses_latest_matching_event(tmp_path: Path) -> None:
    path = tmp_path / "live.log"
    path.write_text(
        "x INFERENCE_MONITOR_HEARTBEAT " + json.dumps({"timestamp": "first"}) + "\n"
        "x OTHER {}\n"
        "x INFERENCE_MONITOR_HEARTBEAT " + json.dumps({"timestamp": "second"}) + "\n",
        encoding="utf-8",
    )
    assert _latest_payload(path, "INFERENCE_MONITOR_HEARTBEAT")["timestamp"] == "second"


def test_close_is_scale_aware_and_fails_on_missing() -> None:
    assert _close(0.00003561, 0.00003561)
    assert _close(1.3688, 1.3688)
    assert not _close(None, 1.0)
    assert not _close(1.0, 1.01)
