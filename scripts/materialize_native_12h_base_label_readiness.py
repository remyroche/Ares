#!/usr/bin/env python3
"""Publish a machine-readable, fail-closed native-12h challenger readiness gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from extreme_price_movements.native_12h_base_label_readiness import build_readiness_gate
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base-oof',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1/oof_predictions.parquet')
 p.add_argument('--native-label-example',type=Path,default=ROOT/'data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels/train_global_long_5_2025_02.parquet')
 p.add_argument('--exact-12h-paths',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1/paths.parquet')
 p.add_argument('--paths-manifest',type=Path,default=ROOT/'data_perp/artifacts/febapr2025_top40_exact1m_paths_20260727_v1/manifest.json')
 p.add_argument('--output-dir',type=Path,default=ROOT/'data_perp/artifacts/native_12h_base_label_challenger_readiness_20260727_v2')
 a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 gate=build_readiness_gate(base_oof=a.base_oof,native_label_example=a.native_label_example,exact_12h_paths=a.exact_12h_paths,paths_manifest=a.paths_manifest)
 a.output_dir.mkdir(parents=True);(a.output_dir/'readiness_gate.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n');print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
