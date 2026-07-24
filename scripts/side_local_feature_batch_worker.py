#!/usr/bin/env python3
"""One bounded static-feature lookup for side-local representation search.

It intentionally runs in a short-lived process because the shared static
feature reader keeps large column buffers while pivoting panels.  Isolating a
small feature batch lets macOS reclaim those buffers before the next batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-parquet", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--features-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.reference_parquet)
    features = json.loads(Path(args.features_json).read_text(encoding="utf-8"))
    values, report = _load_feature_store_columns(
        frame, feature_dir=Path(args.feature_dir), selected_features=list(map(str, features))
    )
    values.to_parquet(args.out, index=False)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"feature_batch rows={len(values)} cols={len(values.columns)}", flush=True)


if __name__ == "__main__":
    main()
