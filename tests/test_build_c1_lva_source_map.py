from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_builds_immutable_target_free_source_map(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame({"__symbol__": ["B/USD:USD", "A/USD:USD", "A/USD:USD"], "final_score": [1.0, 2.0, 3.0]}).to_parquet(source, index=False)
    output = tmp_path / "manifest.json"
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "build_c1_lva_source_map.py"
    subprocess.run([sys.executable, str(runner), "--source", str(source), "--output", str(output)], cwd=root, check=True)
    payload = json.loads(output.read_text())
    assert list(payload["source_map"]) == ["A/USD:USD", "B/USD:USD"]
    assert payload["symbols"] == 2
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, str(runner), "--source", str(source), "--output", str(output)], cwd=root, check=True)
