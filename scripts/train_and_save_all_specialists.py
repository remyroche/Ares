
import asyncio
import time
import pandas as pd
import numpy as np
from src.utils.tprint import tprint_info, tprint_success, tprint_error

# Import all 12 specialists
from src.training.steps.market_analysis.ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
from src.training.steps.market_analysis.ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
from src.training.steps.market_analysis.ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
from src.training.steps.market_analysis.ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
from src.training.steps.market_analysis.ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
from src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
from src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
from src.training.steps.market_analysis.ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
from src.training.steps.market_analysis.ml_spectral_step_enhanced import EnhancedMLSpectralStep
from src.training.steps.market_analysis.ml_microstructure_step_enhanced import EnhancedMLMicrostructureStep
from src.training.steps.market_analysis.xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
from src.training.steps.market_analysis.xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep

async def train_and_save():
    specialists = [
        (EnhancedMLPathRegimeStep, "Path"),
        (EnhancedMLVolumeForceStep, "VolForce"),
        (EnhancedMLRiskRegimeStep, "Risk"),
        (EnhancedMLVolatilityBurstStep, "VolBurst"),
        (EnhancedMLReversionRegimeStep, "Reversion"),
        (EnhancedMLMomentumPersistenceStep, "Momentum"),
        (EnhancedMLLiquidityRegimeStep, "Liquidity"),
        (EnhancedMLSMCRegimeStep, "SMC"),
        (EnhancedMLSpectralStep, "Spectral"),
        (EnhancedMLMicrostructureStep, "Micro"),
        (EnhancedXGBMacroRegimeStep, "Macro"),
        (EnhancedXGBMesoRegimeStep, "Meso")
    ]
    
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'long',
        'is_batch_run': False,  # SAVE ARTIFACTS
        'lookback_days': 1000, # Sufficient data
        'afml_target_sampling_rate': 0.10
    }
    
    results = []
    
    tprint_info(f"🚀 Starting Final Training & Saving of 12 AFML-Hardened Specialists...")
    
    for step_class, name in specialists:
        tprint_info(f"--- Executing {name} ---")
        start = time.time()
        try:
            step = step_class()
            output = await step.execute(config)
            elapsed = time.time() - start
            
            if output.get('success'):
                metrics = output.get('metrics', {})
                results.append({
                    'Name': name,
                    'Success': True,
                    'AUC': f"{metrics.get('auc', 0):.3f}",
                    'MI': f"{metrics.get('mi_score', 0):.4f}",
                    'Samples': output.get('n_samples', 0),
                    'Time': f"{elapsed:.1f}s"
                })
                tprint_success(f"✅ {name} completed and saved: AUC={metrics.get('auc', 0):.3f}, MI={metrics.get('mi_score', 0):.4f}")
            else:
                results.append({'Name': name, 'Success': False, 'Error': output.get('error')})
                tprint_error(f"❌ {name} failed: {output.get('error')}")
        except Exception as e:
            results.append({'Name': name, 'Success': False, 'Error': str(e)})
            tprint_error(f"❌ {name} exception: {e}")
            
    df_results = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL SPECIALIST TRAINING & SAVING REPORT")
    print("="*50)
    print(df_results.to_string(index=False))
    print("="*50)
    
    # Save report
    df_results.to_csv("artifacts/final_specialist_training_report.csv", index=False)

if __name__ == "__main__":
    asyncio.run(train_and_save())
