import hashlib,json
from pathlib import Path
def test_nested_overlay_sealed_and_fail_closed():
 p=Path(__file__).resolve().parents[1]/'data_perp/artifacts/pre2026_nested_residual_context_failure_overlay_20260730_v1'
 assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==(p/'manifest.sha256').read_text().split()[0]
 c=json.loads((p/'contract.json').read_text());assert c['decision_cadence']=='1h' and c['gamma']==.5 and c['ridge_alpha']==30 and c['no_2026']
