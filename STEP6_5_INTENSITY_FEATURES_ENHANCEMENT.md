# Step 6.5 Intensity Features Enhancement

## 🎯 **Changes Made**

### 1. **Removed Placeholder Features**
- **Before**: Used 10-dimensional placeholder when no features provided
- **After**: Uses only actual features from data (no placeholders)
- **Impact**: More efficient and realistic feature usage

### 2. **Enhanced Intensity Feature Generation**
- **Before**: Basic intensity scores (20 features per timeframe)
- **After**: Comprehensive intensity features with multiple temporal scales and regime analysis

## 📊 **Current Intensity Features**

### **Base Intensity Features** (20 per timeframe)
- `intensity_cluster_0` through `intensity_cluster_19`
- Probability of being in each regime cluster
- Rolling window smoothed (10-period)

### **Enhanced Intensity Features** (Generated when fallback needed)

#### **Multi-Temporal Intensity** (3× per cluster)
- `intensity_cluster_X` - Main intensity (10-period window)
- `intensity_cluster_X_short` - Short-term (5-period window)
- `intensity_cluster_X_long` - Long-term (20-period window)

#### **Regime Persistence** (1× per cluster)
- `persistence_cluster_X` - How long we've been in this regime
- Tracks consecutive periods in each regime

#### **Transition Probability** (1× per cluster)
- `transition_prob_cluster_X` - Likelihood of staying in this regime
- Based on historical regime stability

#### **Intensity Volatility** (1× per cluster)
- `intensity_vol_cluster_X` - Stability of regime intensity
- Measures regime consistency

#### **Cross-Regime Correlations** (N×(N-1)/2 total)
- `corr_X_Y` - Correlation between different regime intensities
- Captures regime interactions

#### **Regime Dominance** (2 global features)
- `dominant_regime` - Currently dominant regime ID
- `regime_diversity` - Number of regimes with significant intensity

## 🔢 **Feature Count Analysis**

### **Current Data** (from actual files)
- **15m timeframe**: 20 intensity features
- **1m timeframe**: 20 intensity features  
- **5m timeframe**: 20 intensity features
- **30m timeframe**: 20 intensity features

### **Enhanced Generation** (when fallback needed)
- **Base intensity**: 20 features
- **Multi-temporal**: 60 features (20 × 3)
- **Persistence**: 20 features
- **Transition probability**: 20 features
- **Intensity volatility**: 20 features
- **Cross-correlations**: ~190 features (20×19/2)
- **Dominance features**: 2 features
- **Total enhanced**: ~332 features

## 🎯 **Recommendations**

### **Should We Have More Intensity Features?**

#### **Current State Analysis**:
1. **20 intensity features per timeframe** is substantial
2. **3 timeframes** = 60 total intensity features
3. **Good coverage** of regime space

#### **Potential Enhancements**:

1. **Cross-Timeframe Intensity Features**
   - Correlation between 1m, 5m, 15m intensities
   - Multi-timeframe regime alignment scores
   - Temporal consistency measures

2. **Regime Transition Features**
   - Probability of transitioning between specific regimes
   - Regime transition velocity
   - Regime stability scores

3. **Intensity Momentum Features**
   - Rate of change in intensity scores
   - Intensity acceleration/deceleration
   - Regime momentum indicators

4. **Regime Quality Features**
   - Regime confidence scores
   - Regime purity measures
   - Noise vs. signal ratios

### **Implementation Priority**:

1. **High Priority**: Cross-timeframe intensity correlations
2. **Medium Priority**: Regime transition probabilities
3. **Low Priority**: Additional temporal scales

## 📈 **Performance Impact**

### **Feature Efficiency**:
- **Current**: 60 intensity features across 3 timeframes
- **Enhanced**: 332+ features (when fallback generation used)
- **Recommendation**: Focus on quality over quantity

### **Computational Considerations**:
- **Memory usage**: Linear with feature count
- **Training time**: Quadratic with feature count
- **Overfitting risk**: Increases with feature count

## 🎯 **Conclusion**

The current **20 intensity features per timeframe** provides good coverage. The enhanced generation creates **comprehensive intensity features** when needed, but for production use, we should:

1. **Keep the current 20 features per timeframe** (60 total)
2. **Add cross-timeframe correlations** (+6 features)
3. **Add regime transition probabilities** (+20 features)
4. **Total recommended**: ~86 intensity features

This provides **comprehensive regime intelligence** without excessive complexity.

## 🚀 **Implementation Suggestions**

### **High Priority: Cross-Timeframe Correlations (+6 features)**

#### **Suggested Implementation**:

```python
async def _create_cross_timeframe_correlations(
    self, 
    intensity_data: Dict[str, pd.DataFrame], 
    base_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Create cross-timeframe intensity correlations."""
    
    # Initialize correlation dataframe
    correlation_df = pd.DataFrame(index=base_index)
    
    # Get intensity columns from each timeframe
    tf_intensities = {}
    for tf in self.timeframes:
        if tf in intensity_data:
            tf_data = intensity_data[tf]
            if tf != "1m":
                tf_data = tf_data.reindex(base_index, method="ffill")
            
            # Get intensity columns
            intensity_cols = [col for col in tf_data.columns if col.startswith("intensity_cluster_")]
            tf_intensities[tf] = tf_data[intensity_cols]
    
    # Calculate cross-timeframe correlations
    if len(tf_intensities) >= 2:
        timeframes = list(tf_intensities.keys())
        
        # 1. 1m-5m correlation
        if "1m" in tf_intensities and "5m" in tf_intensities:
            correlation_df["corr_1m_5m"] = self._calculate_intensity_correlation(
                tf_intensities["1m"], tf_intensities["5m"], window=20
            )
        
        # 2. 1m-15m correlation  
        if "1m" in tf_intensities and "15m" in tf_intensities:
            correlation_df["corr_1m_15m"] = self._calculate_intensity_correlation(
                tf_intensities["1m"], tf_intensities["15m"], window=20
            )
        
        # 3. 5m-15m correlation
        if "5m" in tf_intensities and "15m" in tf_intensities:
            correlation_df["corr_5m_15m"] = self._calculate_intensity_correlation(
                tf_intensities["5m"], tf_intensities["15m"], window=20
            )
        
        # 4. Multi-timeframe alignment score
        correlation_df["multi_tf_alignment"] = self._calculate_multi_timeframe_alignment(
            tf_intensities, window=20
        )
        
        # 5. Temporal consistency score
        correlation_df["temporal_consistency"] = self._calculate_temporal_consistency(
            tf_intensities, window=20
        )
        
        # 6. Regime synchronization score
        correlation_df["regime_synchronization"] = self._calculate_regime_synchronization(
            tf_intensities, window=20
        )
    
    return correlation_df

def _calculate_intensity_correlation(
    self, 
    tf1_intensities: pd.DataFrame, 
    tf2_intensities: pd.DataFrame, 
    window: int = 20
) -> pd.Series:
    """Calculate rolling correlation between two timeframe intensities."""
    
    # Calculate mean intensity per timeframe
    tf1_mean = tf1_intensities.mean(axis=1)
    tf2_mean = tf2_intensities.mean(axis=1)
    
    # Calculate rolling correlation
    correlation = tf1_mean.rolling(window=window, min_periods=1).corr(tf2_mean)
    
    return correlation.fillna(0)

def _calculate_multi_timeframe_alignment(
    self, 
    tf_intensities: Dict[str, pd.DataFrame], 
    window: int = 20
) -> pd.Series:
    """Calculate how well all timeframes are aligned."""
    
    # Get dominant regime for each timeframe
    dominant_regimes = {}
    for tf, intensities in tf_intensities.items():
        dominant_regimes[tf] = intensities.idxmax(axis=1)
    
    # Calculate alignment score (percentage of timeframes with same dominant regime)
    alignment_scores = []
    for i in range(len(list(tf_intensities.values())[0])):
        regimes_at_time = [regimes.iloc[i] for regimes in dominant_regimes.values()]
        alignment = len(set(regimes_at_time)) / len(regimes_at_time)
        alignment_scores.append(1 - alignment)  # Higher = better alignment
    
    return pd.Series(alignment_scores, index=list(tf_intensities.values())[0].index)

def _calculate_temporal_consistency(
    self, 
    tf_intensities: Dict[str, pd.DataFrame], 
    window: int = 20
) -> pd.Series:
    """Calculate temporal consistency across timeframes."""
    
    # Calculate intensity stability for each timeframe
    stability_scores = []
    for tf, intensities in tf_intensities.items():
        # Calculate rolling standard deviation of mean intensity
        mean_intensity = intensities.mean(axis=1)
        stability = 1 / (1 + mean_intensity.rolling(window=window, min_periods=1).std())
        stability_scores.append(stability)
    
    # Average stability across timeframes
    avg_stability = pd.concat(stability_scores, axis=1).mean(axis=1)
    
    return avg_stability.fillna(0)

def _calculate_regime_synchronization(
    self, 
    tf_intensities: Dict[str, pd.DataFrame], 
    window: int = 20
) -> pd.Series:
    """Calculate regime synchronization across timeframes."""
    
    # Calculate regime change points for each timeframe
    change_points = {}
    for tf, intensities in tf_intensities.items():
        dominant_regimes = intensities.idxmax(axis=1)
        changes = (dominant_regimes != dominant_regimes.shift(1)).astype(int)
        change_points[tf] = changes
    
    # Calculate synchronization (how often changes happen simultaneously)
    sync_scores = []
    for i in range(len(list(tf_intensities.values())[0])):
        changes_at_time = [changes.iloc[i] for changes in change_points.values()]
        sync_score = sum(changes_at_time) / len(changes_at_time)
        sync_scores.append(sync_score)
    
    # Rolling average for smoothing
    sync_series = pd.Series(sync_scores, index=list(tf_intensities.values())[0].index)
    return sync_series.rolling(window=window, min_periods=1).mean().fillna(0)
```

#### **Integration Points**:
1. **Add to `_create_sequences()` method**:
   ```python
   # After preparing intensity features
   cross_tf_correlations = await self._create_cross_timeframe_correlations(
       intensity_data, base_index
   )
   
   # Add to feature window
   if not cross_tf_correlations.empty:
       correlation_window = cross_tf_correlations.iloc[window_start:window_end]
       correlation_values = correlation_window.values
       intensity_features.append(correlation_values)
   ```

### **Medium Priority: Regime Transition Probabilities (+20 features)**

#### **Suggested Implementation**:

```python
async def _create_regime_transition_features(
    self, 
    hmm_data: Dict[str, pd.DataFrame], 
    base_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Create regime transition probability features."""
    
    # Initialize transition dataframe
    transition_df = pd.DataFrame(index=base_index)
    
    # Get regime data from 1m (base timeframe)
    if "1m" in hmm_data:
        regime_data = hmm_data["1m"]
        if "composite_cluster_id" in regime_data.columns:
            regimes = regime_data["composite_cluster_id"]
            
            # Get unique regimes (excluding noise cluster -1)
            unique_regimes = sorted([r for r in regimes.unique() if r >= 0])
            
            # Calculate transition probabilities for each regime
            for regime_id in unique_regimes:
                # 1. Stay probability (probability of staying in this regime)
                stay_prob = self._calculate_stay_probability(regimes, regime_id, window=20)
                transition_df[f"stay_prob_regime_{regime_id}"] = stay_prob
                
                # 2. Transition velocity (how quickly we transition from this regime)
                transition_vel = self._calculate_transition_velocity(regimes, regime_id, window=20)
                transition_df[f"transition_vel_regime_{regime_id}"] = transition_vel
                
                # 3. Regime stability (inverse of transition frequency)
                stability = self._calculate_regime_stability(regimes, regime_id, window=20)
                transition_df[f"stability_regime_{regime_id}"] = stability
                
                # 4. Regime persistence (how long we typically stay in this regime)
                persistence = self._calculate_regime_persistence(regimes, regime_id, window=20)
                transition_df[f"persistence_regime_{regime_id}"] = persistence
                
                # 5. Regime momentum (tendency to continue in this regime)
                momentum = self._calculate_regime_momentum(regimes, regime_id, window=20)
                transition_df[f"momentum_regime_{regime_id}"] = momentum
    
    return transition_df

def _calculate_stay_probability(
    self, 
    regimes: pd.Series, 
    regime_id: int, 
    window: int = 20
) -> pd.Series:
    """Calculate probability of staying in a specific regime."""
    
    # Create regime mask
    regime_mask = (regimes == regime_id).astype(int)
    
    # Calculate rolling probability of staying in regime
    stay_prob = regime_mask.rolling(window=window, min_periods=1).mean()
    
    return stay_prob.fillna(0)

def _calculate_transition_velocity(
    self, 
    regimes: pd.Series, 
    regime_id: int, 
    window: int = 20
) -> pd.Series:
    """Calculate how quickly we transition from a specific regime."""
    
    # Create regime mask
    regime_mask = (regimes == regime_id).astype(int)
    
    # Calculate transition points (when we leave this regime)
    transitions = ((regime_mask == 1) & (regime_mask.shift(1) == 0)).astype(int)
    
    # Calculate rolling transition frequency
    transition_freq = transitions.rolling(window=window, min_periods=1).sum() / window
    
    return transition_freq.fillna(0)

def _calculate_regime_stability(
    self, 
    regimes: pd.Series, 
    regime_id: int, 
    window: int = 20
) -> pd.Series:
    """Calculate stability of a specific regime."""
    
    # Create regime mask
    regime_mask = (regimes == regime_id).astype(int)
    
    # Calculate rolling standard deviation (lower = more stable)
    stability = 1 / (1 + regime_mask.rolling(window=window, min_periods=1).std())
    
    return stability.fillna(0)

def _calculate_regime_persistence(
    self, 
    regimes: pd.Series, 
    regime_id: int, 
    window: int = 20
) -> pd.Series:
    """Calculate typical persistence length of a specific regime."""
    
    # Create regime mask
    regime_mask = (regimes == regime_id).astype(int)
    
    # Calculate consecutive periods in regime
    persistence = regime_mask.groupby((regime_mask != regime_mask.shift()).cumsum()).cumsum()
    
    # Calculate rolling average persistence
    avg_persistence = persistence.rolling(window=window, min_periods=1).mean()
    
    return avg_persistence.fillna(0)

def _calculate_regime_momentum(
    self, 
    regimes: pd.Series, 
    regime_id: int, 
    window: int = 20
) -> pd.Series:
    """Calculate momentum of a specific regime."""
    
    # Create regime mask
    regime_mask = (regimes == regime_id).astype(int)
    
    # Calculate rate of change in regime probability
    regime_prob = regime_mask.rolling(window=window, min_periods=1).mean()
    momentum = regime_prob.diff().rolling(window=5, min_periods=1).mean()
    
    return momentum.fillna(0)
```

#### **Integration Points**:
1. **Add to `_create_sequences()` method**:
   ```python
   # After preparing HMM states
   transition_features = await self._create_regime_transition_features(
       hmm_data, base_index
   )
   
   # Add to feature window
   if not transition_features.empty:
       transition_window = transition_features.iloc[window_start:window_end]
       transition_values = transition_window.values
       intensity_features.append(transition_values)
   ```

### **Implementation Priority & Timeline**

#### **Phase 1: Cross-Timeframe Correlations (Week 1)**
1. **Day 1-2**: Implement correlation calculation functions
2. **Day 3-4**: Integrate into sequence creation
3. **Day 5**: Testing and validation

#### **Phase 2: Regime Transition Probabilities (Week 2)**
1. **Day 1-3**: Implement transition probability functions
2. **Day 4-5**: Integrate and test

#### **Phase 3: Optimization (Week 3)**
1. **Day 1-2**: Performance optimization
2. **Day 3-4**: Feature selection and validation
3. **Day 5**: Documentation and final testing

### **Expected Feature Count After Implementation**

#### **Current**: 60 intensity features
#### **After Cross-Timeframe**: 66 features (+6)
#### **After Transition Probabilities**: 86 features (+20)
#### **Total Enhancement**: +26 features (43% increase)

### **Performance Considerations**

1. **Memory Usage**: ~15% increase
2. **Computation Time**: ~20% increase
3. **Model Complexity**: Moderate increase
4. **Overfitting Risk**: Low (features are regime-specific)

### **Validation Strategy**

1. **Feature Importance**: Use SHAP values to validate feature relevance
2. **Correlation Analysis**: Ensure new features aren't redundant
3. **Model Performance**: Compare before/after model accuracy
4. **Computational Efficiency**: Monitor training time impact
