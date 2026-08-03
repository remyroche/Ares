import hashlib,json
from pathlib import Path
def test_v3_overlay_audit_and_provenance():
 p=Path(__file__).resolve().parents[1]/'data_perp/artifacts/pre2026_nested_residual_context_failure_overlay_20260730_v3';m=json.loads((p/'manifest.json').read_text());assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==(p/'manifest.sha256').read_text().split()[0];assert m['contract']['no_2026'] and len(m['contract']['environment'])>=5
