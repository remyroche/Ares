import asyncio
import pandas as pd
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.config import CONFIG
from src.training.steps.step2_market_regime_classification import convert_trade_data_to_ohlcv

async def test_force_retraining():
    print("=== Testing Force Retraining of Regime Classifier ===")
    
    # Load data
    df = pd.read_parquet('data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet')
    ohlcv = convert_trade_data_to_ohlcv(df, '1h')
    print(f"OHLCV shape: {ohlcv.shape}")
    
    # Create classifier with force retraining
    classifier = UnifiedRegimeClassifier(CONFIG, 'BINANCE', 'ETHUSDT')
    
    # Force retraining by deleting any existing models
    import os
    import glob
    
    # Delete any existing model files
    model_patterns = [
        "checkpoints/analyst_models/*.joblib",
        "checkpoints/analyst_models/*.h5",
        "models/*.pkl",
        "models/*.joblib"
    ]
    
    for pattern in model_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except:
                pass
    
    # Initialize without loading existing models
    await classifier.initialize()
    
    # Force retraining by calling train methods directly
    print("\n=== Training HMM Model ===")
    hmm_result = await classifier.train_hmm_labeler(ohlcv)
    print(f"HMM training result: {hmm_result}")
    
    print("\n=== Training Ensemble Model ===")
    ensemble_result = await classifier.train_basic_ensemble(ohlcv)
    print(f"Ensemble training result: {ensemble_result}")
    
    print("\n=== Training Location Classifier ===")
    location_result = await classifier.train_location_classifier(ohlcv)
    print(f"Location training result: {location_result}")
    
    # Test classification
    print("\n=== Testing Classification ===")
    result = await classifier.classify_regimes(ohlcv)
    print(f"Classification result: {result}")
    
    # Check regime distribution
    if 'regime_distribution' in result:
        print(f"\nRegime Distribution: {result['regime_distribution']}")
        distinct_regimes = len(result['regime_distribution'])
        print(f"Distinct regimes found: {distinct_regimes}")
        
        if distinct_regimes == 1:
            print("⚠️  WARNING: Only 1 distinct regime found!")
            print("This suggests the ADX threshold might still be too high.")
        else:
            print("✅ Multiple regimes found!")

if __name__ == "__main__":
    asyncio.run(test_force_retraining()) 