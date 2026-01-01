#!/usr/bin/env python3
"""Update all specialist steps with independent diagnostics."""

import os
import re

def update_specialist_step(filepath):
    """Update a single specialist step with diagnostics mixin."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if already updated
        if 'SpecialistDiagnosticsMixin' in content:
            print(f"✅ {os.path.basename(filepath)} already updated")
            return True
        
        # Add import for diagnostics mixin
        import_pattern = r'(from src\.training\.steps\.market_analysis\.ml_risk_regime_step import MLRiskRegimeStepHMM)'
        if import_pattern in content:
            content = re.sub(
                import_pattern,
                r'\1\nfrom src.training.steps.market_analysis.specialist_diagnostics_mixin import SpecialistDiagnosticsMixin',
                content
            )
        
        # Update class inheritance
        class_pattern = r'class (\w+)\(MLRiskRegimeStepHMM\):'
        content = re.sub(class_pattern, r'class \1(SpecialistDiagnosticsMixin, MLRiskRegimeStepHMM):', content)
        
        # Add diagnostics method at the end
        if content.endswith('}\n'):
            class_name = re.search(r'class (\w+)\(', content).group(1)
            diagnostics_method = f'''
    def run_diagnostics(self, symbol: str = 'ETHUSDT', exchange: str = 'binance', 
                       timeframe: str = '15m', direction: str = 'long') -> Dict[str, Any]:
        """Run independent diagnostics for {class_name.replace('Step', '').replace('ML', '').lower()} specialist."""
        return self.run_self_diagnostics(symbol, exchange, timeframe, direction)
'''
            content = content[:-2] + diagnostics_method + '\n}\n'
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {os.path.basename(filepath)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {os.path.basename(filepath)}: {e}")
        return False

# List of specialist files to update
specialist_files = [
    'src/training/steps/market_analysis/ml_volatility_burst_step.py',
    'src/training/steps/market_analysis/ml_risk_regime_step.py',
    'src/training/steps/market_analysis/ml_liquidity_regime_step.py',
    'src/training/steps/market_analysis/ml_breakout_bounce_regime_step.py',
    'src/training/steps/market_analysis/ml_path_regime_step.py',
    'src/training/steps/market_analysis/ml_reversion_regime_step.py',
    'src/training/steps/market_analysis/ml_smc_regime_step.py',
    'src/training/steps/market_analysis/ml_volume_force_step.py',
]

print("🔧 Updating specialist steps with independent diagnostics...")
success_count = 0
for filepath in specialist_files:
    if os.path.exists(filepath):
        if update_specialist_step(filepath):
            success_count += 1
    else:
        print(f"⚠️ File not found: {filepath}")

print(f"\n✅ Successfully updated {success_count}/{len(specialist_files)} specialist steps")
