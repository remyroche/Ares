# Feature Engineering Integration with HMM Regime Optimization

## 🎯 **Overview**

Feature engineering integration allows the optimization system to automatically select, create, and optimize features alongside HMM regime parameters, creating a comprehensive end-to-end optimization pipeline.

## 🔧 **Integration Approaches**

### **1. Hierarchical Optimization**
```python
# Two-stage optimization
# Stage 1: Optimize feature engineering
# Stage 2: Optimize HMM parameters with best features

def hierarchical_optimization(data):
    # Feature engineering optimization
    best_features = optimize_feature_engineering(data)
    
    # HMM optimization with best features
    best_params = optimize_hmm_parameters(data, best_features)
    
    return best_features, best_params
```

### **2. Joint Optimization**
```python
# Single-stage optimization of both features and HMM parameters
def joint_optimization(data):
    def objective(trial):
        # Suggest feature engineering parameters
        feature_params = suggest_feature_params(trial)
        
        # Suggest HMM parameters
        hmm_params = suggest_hmm_params(trial)
        
        # Apply feature engineering
        engineered_features = apply_feature_engineering(data, feature_params)
        
        # Apply HMM optimization
        score = evaluate_hmm_regimes(engineered_features, hmm_params)
        
        return score
    
    return optimize(objective)
```

### **3. Adaptive Feature Selection**
```python
# Dynamically select features based on regime quality
def adaptive_feature_selection(data, base_features):
    feature_importance = {}
    
    for feature in base_features:
        # Test feature individually
        score = evaluate_single_feature(data, feature)
        feature_importance[feature] = score
    
    # Select top features
    top_features = select_top_features(feature_importance, threshold=0.7)
    
    return top_features
```

## 📊 **Feature Engineering Components**

### **1. Feature Selection Methods**

#### **Variance-Based Selection**
```python
def variance_feature_selection(data, threshold=0.01):
    """Select features based on variance threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selected_features = selector.fit_transform(data)
    return selected_features, selector.get_support()
```

#### **Correlation-Based Selection**
```python
def correlation_feature_selection(data, threshold=0.95):
    """Remove highly correlated features."""
    corr_matrix = data.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    return data.drop(columns=to_drop)
```

#### **Statistical Feature Selection**
```python
def statistical_feature_selection(data, target, k=10):
    """Select features using statistical tests."""
    selector = SelectKBest(score_func=f_regression, k=k)
    selected_features = selector.fit_transform(data, target)
    return selected_features, selector.get_support()
```

### **2. Feature Creation Methods**

#### **Technical Indicators**
```python
def create_technical_indicators(data):
    """Create technical indicators from price data."""
    features = {}
    
    # Moving averages
    features['sma_20'] = data['close'].rolling(20).mean()
    features['sma_50'] = data['close'].rolling(50).mean()
    features['ema_12'] = data['close'].ewm(span=12).mean()
    
    # Momentum indicators
    features['rsi_14'] = calculate_rsi(data['close'], 14)
    features['macd'] = calculate_macd(data['close'])
    features['stoch_k'] = calculate_stochastic(data, 14)
    
    # Volatility indicators
    features['bb_upper'] = calculate_bollinger_bands(data['close'], 20, 2)[0]
    features['bb_lower'] = calculate_bollinger_bands(data['close'], 20, 2)[1]
    features['atr_14'] = calculate_atr(data, 14)
    
    return pd.DataFrame(features)
```

#### **Market Microstructure Features**
```python
def create_microstructure_features(data):
    """Create market microstructure features."""
    features = {}
    
    # Volume-based features
    features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['volume_price_trend'] = calculate_vpt(data)
    features['money_flow_index'] = calculate_mfi(data, 14)
    
    # Price-based features
    features['price_momentum'] = data['close'].pct_change(10)
    features['price_acceleration'] = data['close'].pct_change().diff()
    features['price_volatility'] = data['close'].pct_change().rolling(20).std()
    
    # Spread and liquidity features
    features['bid_ask_spread'] = data['ask'] - data['bid']
    features['spread_ratio'] = features['bid_ask_spread'] / data['close']
    
    return pd.DataFrame(features)
```

#### **Regime-Specific Features**
```python
def create_regime_features(data, regime_labels):
    """Create features specific to regime characteristics."""
    features = {}
    
    # Regime transition features
    features['regime_duration'] = calculate_regime_duration(regime_labels)
    features['regime_transition_prob'] = calculate_transition_probabilities(regime_labels)
    
    # Regime-specific statistics
    for regime in np.unique(regime_labels):
        regime_mask = regime_labels == regime
        regime_data = data[regime_mask]
        
        features[f'regime_{regime}_volatility'] = regime_data['close'].pct_change().std()
        features[f'regime_{regime}_volume_mean'] = regime_data['volume'].mean()
        features[f'regime_{regime}_returns_mean'] = regime_data['close'].pct_change().mean()
    
    return pd.DataFrame(features)
```

### **3. Feature Transformation Methods**

#### **Dimensionality Reduction**
```python
def apply_dimensionality_reduction(data, method='pca', n_components=None):
    """Apply dimensionality reduction techniques."""
    
    if method == 'pca':
        if n_components is None:
            n_components = min(data.shape[1], 20)
        
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        
        return reduced_data, pca
    
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=n_components, random_state=42)
        reduced_data = ica.fit_transform(data)
        
        return reduced_data, ica
    
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_data = tsne.fit_transform(data)
        
        return reduced_data, tsne
```

#### **Feature Scaling**
```python
def apply_feature_scaling(data, method='standard'):
    """Apply feature scaling methods."""
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='normal')
    
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
```

## 🔄 **Integration with Optimization**

### **1. Feature Engineering Parameters**

```python
def suggest_feature_engineering_params(trial):
    """Suggest feature engineering parameters for optimization."""
    
    return {
        # Feature selection
        'feature_selection_method': trial.suggest_categorical(
            'feature_selection_method', 
            ['variance', 'correlation', 'statistical', 'none']
        ),
        'variance_threshold': trial.suggest_float('variance_threshold', 0.001, 0.1),
        'correlation_threshold': trial.suggest_float('correlation_threshold', 0.7, 0.99),
        'n_features_select': trial.suggest_int('n_features_select', 5, 50),
        
        # Feature creation
        'use_technical_indicators': trial.suggest_categorical('use_technical_indicators', [True, False]),
        'use_microstructure_features': trial.suggest_categorical('use_microstructure_features', [True, False]),
        'use_regime_features': trial.suggest_categorical('use_regime_features', [True, False]),
        
        # Dimensionality reduction
        'use_dimensionality_reduction': trial.suggest_categorical('use_dimensionality_reduction', [True, False]),
        'reduction_method': trial.suggest_categorical('reduction_method', ['pca', 'ica', 'tsne']),
        'n_components': trial.suggest_int('n_components', 5, 30),
        
        # Scaling
        'scaling_method': trial.suggest_categorical('scaling_method', ['standard', 'robust', 'minmax', 'quantile']),
        
        # Feature interaction
        'use_feature_interactions': trial.suggest_categorical('use_feature_interactions', [True, False]),
        'interaction_degree': trial.suggest_int('interaction_degree', 2, 3),
    }
```

### **2. Integrated Objective Function**

```python
def create_integrated_objective(data, base_features):
    """Create objective function that optimizes both features and HMM parameters."""
    
    def objective(trial):
        # Suggest feature engineering parameters
        feature_params = suggest_feature_engineering_params(trial)
        
        # Suggest HMM parameters
        hmm_params = suggest_hmm_params(trial)
        
        try:
            # Apply feature engineering
            engineered_data = apply_feature_engineering_pipeline(data, feature_params)
            
            # Apply HMM optimization
            cluster_data = generate_hmm_clusters(engineered_data, hmm_params)
            
            # Evaluate regime quality
            score = evaluate_regime_quality(cluster_data, engineered_data)
            
            return score
            
        except Exception as e:
            return -np.inf
    
    return objective
```

### **3. Feature Engineering Pipeline**

```python
def apply_feature_engineering_pipeline(data, params):
    """Apply complete feature engineering pipeline."""
    
    result_data = data.copy()
    
    # 1. Feature selection
    if params['feature_selection_method'] != 'none':
        result_data = apply_feature_selection(result_data, params)
    
    # 2. Feature creation
    if params['use_technical_indicators']:
        tech_features = create_technical_indicators(data)
        result_data = pd.concat([result_data, tech_features], axis=1)
    
    if params['use_microstructure_features']:
        micro_features = create_microstructure_features(data)
        result_data = pd.concat([result_data, micro_features], axis=1)
    
    # 3. Feature interactions
    if params['use_feature_interactions']:
        interaction_features = create_feature_interactions(result_data, params['interaction_degree'])
        result_data = pd.concat([result_data, interaction_features], axis=1)
    
    # 4. Dimensionality reduction
    if params['use_dimensionality_reduction']:
        result_data, reducer = apply_dimensionality_reduction(
            result_data, params['reduction_method'], params['n_components']
        )
    
    # 5. Scaling
    result_data, scaler = apply_feature_scaling(result_data, params['scaling_method'])
    
    return result_data
```

## 📈 **Advanced Feature Engineering**

### **1. Adaptive Feature Engineering**

```python
class AdaptiveFeatureEngineer:
    """Adaptive feature engineering that learns from optimization results."""
    
    def __init__(self):
        self.feature_importance_history = []
        self.best_feature_combinations = []
    
    def update_feature_importance(self, trial_results):
        """Update feature importance based on trial results."""
        
        for trial in trial_results:
            if trial['score'] > 0.7:  # Good trials
                feature_params = trial['feature_params']
                self.feature_importance_history.append({
                    'params': feature_params,
                    'score': trial['score']
                })
    
    def suggest_improved_features(self, trial):
        """Suggest features based on historical performance."""
        
        if len(self.feature_importance_history) > 10:
            # Analyze successful feature combinations
            successful_features = self._analyze_successful_features()
            
            # Suggest similar combinations
            return self._suggest_similar_features(trial, successful_features)
        else:
            # Use default suggestions
            return suggest_feature_engineering_params(trial)
```

### **2. Feature Evolution**

```python
def evolve_features(data, generations=10, population_size=20):
    """Evolve features using genetic algorithms."""
    
    from deap import base, creator, tools, algorithms
    import random
    
    # Define genetic algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define genes (feature engineering parameters)
    toolbox.register("feature_params", suggest_feature_engineering_params)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.feature_params, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define genetic operators
    toolbox.register("evaluate", evaluate_feature_combination, data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution
    population = toolbox.population(n=population_size)
    
    for gen in range(generations):
        offspring = map(toolbox.clone, population)
        offspring = list(offspring)
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
    
    return tools.selBest(population, 1)[0]
```

## 🎯 **Best Practices**

### **1. Feature Engineering Strategy**

- **Start Simple**: Begin with basic technical indicators
- **Iterate**: Add complexity based on performance
- **Validate**: Use cross-validation for feature selection
- **Monitor**: Track feature importance over time

### **2. Performance Optimization**

- **Caching**: Cache expensive feature calculations
- **Parallelization**: Use parallel processing for feature creation
- **Memory Management**: Stream large datasets
- **Early Stopping**: Stop unpromising feature combinations

### **3. Quality Assurance**

- **Feature Stability**: Ensure features are stable over time
- **Outlier Handling**: Handle outliers in feature creation
- **Missing Data**: Implement robust missing data strategies
- **Feature Drift**: Monitor for feature drift in production

## 📊 **Monitoring and Evaluation**

### **1. Feature Performance Tracking**

```python
def track_feature_performance(optimization_history):
    """Track feature performance across optimization trials."""
    
    feature_performance = {}
    
    for trial in optimization_history:
        feature_params = trial['feature_params']
        score = trial['score']
        
        for param_name, param_value in feature_params.items():
            if param_name not in feature_performance:
                feature_performance[param_name] = []
            
            feature_performance[param_name].append({
                'value': param_value,
                'score': score
            })
    
    return feature_performance
```

### **2. Feature Importance Analysis**

```python
def analyze_feature_importance(feature_performance):
    """Analyze feature importance based on performance."""
    
    importance_scores = {}
    
    for feature_name, performance_data in feature_performance.items():
        # Calculate correlation between feature values and scores
        values = [p['value'] for p in performance_data]
        scores = [p['score'] for p in performance_data]
        
        if len(set(values)) > 1:  # Feature varies
            correlation = np.corrcoef(values, scores)[0, 1]
            importance_scores[feature_name] = abs(correlation)
    
    return importance_scores
```

This comprehensive feature engineering integration ensures that the HMM regime optimization system can automatically discover and optimize the best feature combinations for regime discovery, leading to more robust and effective trading strategies.