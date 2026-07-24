#!/usr/bin/env python3
"""Export a frozen AE/GMM state's ordered input contract as CSV."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    with args.state_path.open("rb") as handle:
        state = pickle.load(handle)
    features = [str(value) for value in state.get("feature_columns", []) if str(value)]
    if len(features) < 2:
        raise ValueError("Frozen AE/GMM state has no usable input contract")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": features}).to_csv(args.output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
