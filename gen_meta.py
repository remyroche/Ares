import sys
import os
sys.path.insert(0, os.getcwd())
from extreme_price_movements.config import CFG
from extreme_price_movements.reports.bucket_report import report_meta_training

cfg = CFG
run_id = cfg.get("run_id", "20260214_190000")
data_root = cfg.get("data_root", "data")
reports_root = cfg.get("reports_root", "reports")

try:
    rp = report_meta_training(run_id, data_root, {}, cfg, base_dir=reports_root)
    print(f"Generated report at: {rp}")
except Exception as e:
    import traceback
    traceback.print_exc()
