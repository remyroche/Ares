# 1-Hour Timeframe Regime Analysis Implementation

## Rationale
Using only 1h timeframe for regime analysis provides:
- Sufficient data points for statistical significance
- Filters out market noise from shorter timeframes
- Captures meaningful market structure changes
- Balances between responsiveness and stability

## Implementation in Step 3 (HMM Regime Discovery)

```python
class HMMRegimeDiscoveryStep:
    def __init__(self, config):
        self.config = config
        # Force 1h timeframe for regime analysis
        self.regime_timeframe = '1h'
        self.logger = system_logger.getChild("HMMRegimeDiscoveryStep")
        
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str):
        """Execute HMM regime discovery using 1h data regardless of trading timeframe."""
        
        # Load 1h data for regime analysis
        regime_data = await self._load_1h_data(symbol, exchange, data_dir)
        
        # Perform regime discovery on 1h data
        regime_model = await self._discover_regimes_1h(regime_data)
        
        # Map 1h regimes back to original timeframe
        if timeframe != '1h':
            regime_labels = await self._map_regimes_to_timeframe(
                regime_model, 
                regime_data,
                target_timeframe=timeframe
            )
        else:
            regime_labels = regime_model.labels_
            
        return {
            'regime_model': regime_model,
            'regime_labels': regime_labels,
            'regime_timeframe': '1h',
            'trading_timeframe': timeframe
        }
    
    async def _load_1h_data(self, symbol, exchange, data_dir):
        """Load or resample to 1h data."""
        # Try to load existing 1h data
        path_1h = f"{data_dir}/{exchange}_{symbol}_1h_unified.parquet"
        
        if os.path.exists(path_1h):
            return pd.read_parquet(path_1h)
        else:
            # Resample from lower timeframe
            self.logger.info("Resampling data to 1h for regime analysis...")
            lower_tf_data = await self._load_lowest_timeframe(symbol, exchange, data_dir)
            return self._resample_to_1h(lower_tf_data)
    
    def _resample_to_1h(self, data):
        """Resample OHLCV data to 1h."""
        return data.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    async def _map_regimes_to_timeframe(self, regime_model, regime_data_1h, target_timeframe):
        """Map 1h regime labels to target timeframe."""
        # Create mapping based on timestamps
        regime_mapping = pd.DataFrame({
            'timestamp': regime_data_1h.index,
            'regime': regime_model.labels_
        })
        
        # Forward-fill regimes to target timeframe
        target_data = await self._load_target_timeframe_data(target_timeframe)
        
        # Merge and forward-fill
        merged = pd.merge_asof(
            target_data[['timestamp']], 
            regime_mapping,
            on='timestamp',
            direction='backward'
        )
        
        return merged['regime'].values
```

## Configuration Update

```yaml
# config/regime_config.yaml
regime_discovery:
  # Fixed 1h timeframe for all regime analysis
  analysis_timeframe: "1h"
  
  # Minimum data requirements for 1h
  min_hours_required: 720  # 30 days of 1h data
  
  # HMM parameters optimized for 1h
  hmm_params:
    n_components: 4
    covariance_type: "full"
    n_iter: 100
    
  # Regime mapping settings
  mapping:
    method: "backward_fill"  # Use most recent 1h regime
    validate_alignment: true
```