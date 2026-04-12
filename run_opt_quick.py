#!/usr/bin/env python3
from extreme_price_movements.policy_optimiser import run_policy_optimisation

result = run_policy_optimisation(
    data_root='data',
    run_id='20260321_140000',
    holdout_frac=0.30,
    cost_pct=0.003,
    use_offset_optimiser=False,
)

print()
print('='*60)
print('RESULTS:')
print('='*60)
for s in result.get('strategies', []):
    sid = s.get('strategy_id', 'N/A')[:45]
    val_base = s.get('baseline_net_pnl', 0)
    val_final = s.get('final_net_pnl', 0)
    val_delta = s.get('net_pnl_delta', 0)
    full_base = s.get('baseline_net_pnl_full', val_base)
    full_final = s.get('final_net_pnl_full', val_final)
    full_delta = s.get('net_pnl_delta_full', val_delta)
    reverted = '✓ REVERTED' if s.get('_reverted_to_baseline_') else ''
    print(f'{sid}...')
    print(f'  Validation: {val_base:+.4f} → {val_final:+.4f} (Δ{val_delta:+.4f}) {reverted}')
    print(f'  Full:       {full_base:+.4f} → {full_final:+.4f} (Δ{full_delta:+.4f})')
