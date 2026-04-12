#!/usr/bin/env python3
"""Diagnose policy optimization for profitable strategy."""

import sys
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from extreme_price_movements.policy_optimiser import run_policy_optimisation

result = run_policy_optimisation(
    data_root='data',
    run_id='20260321_140000',
    holdout_frac=0.30,
    cost_pct=0.003,
    use_offset_optimiser=False,
)

print()
print('='*80)
print('OPTIMIZATION DIAGNOSTICS')
print('='*80)

if result and result.get('strategies'):
    # Find the profitable strategy that reverted to baseline
    for s in result['strategies']:
        if s.get('baseline_net_pnl', 0) > 10 and s.get('net_pnl_delta', 0) == 0:
            print(f"\nStrategy: {s.get('strategy_id', 'N/A')[:60]}...")
            print(f"Baseline PnL: {s.get('baseline_net_pnl', 0):+.6f}")
            print(f"Final PnL: {s.get('final_net_pnl', 0):+.6f}")
            print(f"Reverted to baseline: {s.get('final_net_pnl', 0) == s.get('baseline_net_pnl', 0)}")
            
            history = s.get('_param_history_', [])
            if history:
                print(f"\nParameter optimization path ({len(history)} steps):")
                print(f"{'Step':<5} {'Parameter':<25} {'Value':<15} {'Train PnL':<12} {'Val PnL':<12} {'Delta':<10}")
                print("-" * 85)
                baseline_val = s.get('baseline_net_pnl', 0)
                for i, (name, value, train_pnl, val_pnl) in enumerate(history[:20]):  # First 20 steps
                    delta = val_pnl - baseline_val
                    status = "✓" if delta > 0 else "✗" if delta < 0 else "→"
                    value_str = str(value)[:14]
                    print(f"{i+1:<5} {name:<25} {value_str:<15} {train_pnl:<12.4f} {val_pnl:<12.4f} {delta:+.4f} {status}")
            break
