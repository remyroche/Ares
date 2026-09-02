from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_strict_r3_single_head_downstream_source_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("single_head_downstream_source", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_modern_score_subroot_is_explicit_and_does_not_change_single_head_semantics() -> None:
    source = SCRIPT.read_text()
    assert "--score-subroot" in source
    assert "--upstream-coordinate" in source
    assert 'args.single_head_root / str(args.score_subroot) / f"month={month_key}.parquet"' in source
    assert '"coordinate": str(args.upstream_coordinate)' in source
    assert "all four historical score slots equal the one head score" in source
