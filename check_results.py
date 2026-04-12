#!/usr/bin/env python3
import json

with open('data/artifacts/20260321_140000/policy_params/strategy_final_acceptation.json') as f:
    data = json.load(f)

print('='*60)
print('FINAL ACCEPTATION RESULTS')
print('='*60)
for s in data.get('strategies', []):
    sid = s.get('strategy_id', 'N/A')[:50]
    baseline = s.get('baseline_net_pnl', 0)
    final = s.get('final_net_pnl', 0)
    delta = s.get('net_pnl_delta', 0)
    status = '✓' if delta > 0.001 else '→' if abs(delta) < 0.001 else '✗'
    print(f'{sid}...')
    print(f'  {status} Baseline: {baseline:+.4f} | Final: {final:+.4f} | Delta: {delta:+.4f}')
    if s.get('_reverted_to_baseline_'):
        print('  (Reverted to baseline - optimization degraded)')
    if 'tp_mult' in s:
        print(f'  TP={s.get("tp_mult", "N/A"):.2f} SL={s.get("sl_mult", "N/A"):.2f}')
    if 'trail_activation_atr' in s:
        print(f'  Trail: act={s.get("trail_activation_atr"):.2f} give={s.get("trail_giveback_atr"):.2f}')
