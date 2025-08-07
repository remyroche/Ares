import asyncio
from src.training.steps.step2_market_regime_classification import MarketRegimeClassificationStep
from src.config import CONFIG

async def test_step2():
    step = MarketRegimeClassificationStep(CONFIG)
    await step.initialize()
    
    training_input = {
        'symbol': 'ETHUSDT', 
        'exchange': 'BINANCE', 
        'data_dir': 'data/training'
    }
    pipeline_state = {}
    
    result = await step.execute(training_input, pipeline_state)
    print(f'Step result: {result}')
    
    # Check if the regime file was created
    import os
    regime_file = 'data/training/BINANCE_ETHUSDT_regime_classification.json'
    if os.path.exists(regime_file):
        print(f"✅ Regime file created: {regime_file}")
        import json
        with open(regime_file, 'r') as f:
            data = json.load(f)
            print(f"Regime file content: {data}")
    else:
        print(f"❌ Regime file not found: {regime_file}")

if __name__ == "__main__":
    asyncio.run(test_step2()) 