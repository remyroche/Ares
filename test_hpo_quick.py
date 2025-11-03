"""Quick test script for HPO system - bypasses slow launcher."""

import asyncio
import sys
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

async def test_hpo():
    print("="*80)
    print("QUICK HPO TEST - Bypassing Launcher")
    print("="*80)
    
    from src.training.steps.model_training.unified_models_training_step import UnifiedModelsTrainingStep
    
    step = UnifiedModelsTrainingStep()
    
    config = {
        'training_type': 'analyst_base',
        'symbol': 'ETHUSDT',
        'timeframe': '15m',
        'direction': 'long',
        'exchange': 'binance',
        'execution_mode': 'light',
        'enable_hpo': True
    }
    
    print(f"\nRunning with config: {config}")
    print("\nStarting training (HPO enabled)...\n")
    
    result = await step.execute(config)
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Success: {result.get('success')}")
    
    if not result.get('success'):
        print(f"\nError: {result.get('error')}")
        if 'traceback' in result:
            print(f"\nTraceback:\n{result['traceback']}")
    else:
        print(f"\nArtifacts created: {list(result.get('artifacts', {}).keys())}")
        print(f"Metrics: {result.get('metrics', {})}")
    
    print("="*80)

if __name__ == '__main__':
    asyncio.run(test_hpo())

