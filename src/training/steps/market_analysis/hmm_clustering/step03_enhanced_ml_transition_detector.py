import pandas as pd
import numpy as np

'Enhanced ML-Based Regime Transition Detection with Random Forest + LGBM.\n\nThis module implements the specific approach requested:\n1. Random Forest for feature selection (feature importance + permutation importance)\n2. LGBM iterative selection (starting with top 20 features, adding 10 at a time)\n3. Stop when performance plateaus or decreases\n'
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
import json
import typing

warnings.filterwarnings('ignore')

class EnhancedMLRegimeTransitionDetector:
    """Enhanced ML-based regime transition detector with Random Forest + LGBM."""

    def __init__(self, config: Dict[str, Any]=None) -> None:
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        self.initial_features = self.config.get('initial_features', 20)
        self.feature_increment = self.config.get('feature_increment', 10)
        self.max_features = self.config.get('max_features', 100)
        self.min_improvement = self.config.get('min_improvement', 0.001)
        self.patience = self.config.get('patience', 3)
        self.rf_params = self.config.get('rf_params', {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1})
        self.lgb_params = self.config.get('lgb_params', {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42})
        self.feature_importance = {}
        self.permutation_importance = {}
        self.selected_features = []
        self.best_lgb_model = None
        self.best_performance = 0.0
        self.feature_selection_history = []

    def train_transition_models(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """
        Train ML models to detect regime transitions using Random Forest + LGBM approach.
        
        Args:
            data: Market data with OHLCV columns
            regimes: Regime labels for each data point
            
        Returns:
            Training results and model performance
        """
        training_results = {'feature_selection_completed': False, 'lgb_training_completed': False, 'feature_importance': {}, 'permutation_importance': {}, 'selected_features': [], 'best_performance': 0.0, 'feature_selection_history': [], 'training_summary': {}}
        try:
            X, y = self._prepare_transition_training_data(data, regimes)
            if len(X) == 0 or len(np.unique(y)) < 2:
                return {'error': 'Insufficient data for training transition models'}
            rf_results = self._random_forest_feature_selection(X, y)
            training_results['feature_importance'] = rf_results['feature_importance']
            training_results['permutation_importance'] = rf_results['permutation_importance']
            training_results['feature_selection_completed'] = True
            lgb_results = self._lgbm_iterative_feature_selection(X, y, rf_results['combined_importance'])
            training_results['selected_features'] = lgb_results['selected_features']
            training_results['best_performance'] = lgb_results['best_performance']
            training_results['feature_selection_history'] = lgb_results['selection_history']
            training_results['lgb_training_completed'] = True
            final_model_results = self._train_final_lgbm_model(X, y, lgb_results['selected_features'])
            training_results['final_model'] = final_model_results['model']
            training_results['final_performance'] = final_model_results['performance']
            training_results['final_scaler'] = final_model_results['scaler']
            training_results['training_summary'] = self._generate_training_summary(training_results)
            return training_results
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}

    def _prepare_transition_training_data(self, data: pd.DataFrame, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for regime transition detection."""
        transition_labels = self._create_transition_labels(regimes)
        features = self._create_comprehensive_transition_features(data, regimes)
        min_length = min(len(features), len(transition_labels))
        features = features[:min_length]
        transition_labels = transition_labels[:min_length]
        max_lag = 20
        valid_mask = np.arange(len(features)) >= max_lag
        features = features[valid_mask]
        transition_labels = transition_labels[valid_mask]
        transition_labels = self._handle_class_imbalance(transition_labels)
        return (features, transition_labels)

    def _create_transition_labels(self, regimes: np.ndarray) -> np.ndarray:
        """Create labels for regime transitions."""
        transition_labels = np.zeros(len(regimes), dtype=int)
        prediction_horizon = 3
        for i in range(len(regimes) - prediction_horizon):
            current_regime = regimes[i]
            future_regimes = regimes[i + 1:i + prediction_horizon + 1]
            if np.any(future_regimes != current_regime):
                transition_labels[i] = 1
        return transition_labels

    def _create_comprehensive_transition_features(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Create comprehensive features for transition prediction."""
        feature_list = []
        price_features = self._create_price_transition_features(data)
        feature_list.append(price_features)
        volume_features = self._create_volume_transition_features(data)
        feature_list.append(volume_features)
        volatility_features = self._create_volatility_transition_features(data)
        feature_list.append(volatility_features)
        regime_features = self._create_regime_transition_features(regimes)
        feature_list.append(regime_features)
        technical_features = self._create_technical_transition_features(data)
        feature_list.append(technical_features)
        interaction_features = self._create_interaction_transition_features(data, regimes)
        feature_list.append(interaction_features)
        lagged_features = self._create_lagged_features(data, regimes)
        feature_list.append(lagged_features)
        all_features = np.concatenate([f for f in feature_list if f is not None], axis=1)
        return all_features

    def _create_price_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create price-based transition features."""
        features = []
        for window in [1, 2, 3, 5, 10, 20]:
            momentum = data['close'].pct_change(window)
            features.append(momentum.values.reshape(-1, 1))
            momentum_acc = momentum.diff()
            features.append(momentum_acc.values.reshape(-1, 1))
        for window in [10, 20, 50]:
            rolling_high = data['high'].rolling(window).max()
            rolling_low = data['low'].rolling(window).min()
            price_position = (data['close'] - rolling_low) / (rolling_high - rolling_low)
            features.append(price_position.values.reshape(-1, 1))
        price_range = (data['high'] - data['low']) / data['close']
        features.append(price_range.values.reshape(-1, 1))
        price_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features.append(price_gap.values.reshape(-1, 1))
        for window in [5, 10, 20]:
            price_vol = data['close'].pct_change().rolling(window).std()
            features.append(price_vol.values.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _create_volume_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create volume-based transition features."""
        features = []
        for window in [1, 2, 3, 5, 10, 20]:
            volume_momentum = data['volume'].pct_change(window)
            features.append(volume_momentum.values.reshape(-1, 1))
        for window in [5, 10, 20, 50]:
            volume_ratio = data['volume'] / data['volume'].rolling(window).mean()
            features.append(volume_ratio.values.reshape(-1, 1))
        volume_volatility = data['volume'].rolling(20).std() / data['volume'].rolling(20).mean()
        features.append(volume_volatility.values.reshape(-1, 1))
        volume_price_trend = (data['close'].pct_change() * data['volume']).rolling(10).sum()
        features.append(volume_price_trend.values.reshape(-1, 1))
        volume_spikes = (data['volume'] > data['volume'].rolling(20).mean() + 2 * data['volume'].rolling(20).std()).astype(int)
        features.append(volume_spikes.values.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _create_volatility_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create volatility-based transition features."""
        features = []
        for window in [5, 10, 20, 50]:
            volatility = data['close'].pct_change().rolling(window).std()
            features.append(volatility.values.reshape(-1, 1))
            vol_momentum = volatility.pct_change()
            features.append(vol_momentum.values.reshape(-1, 1))
        returns = data['close'].pct_change()
        vol_of_vol = returns.rolling(20).std().rolling(10).std()
        features.append(vol_of_vol.values.reshape(-1, 1))
        vol_regime = self._classify_volatility_regime(returns.rolling(20).std())
        features.append(vol_regime.values.reshape(-1, 1))
        vol_clustering = volatility.rolling(50).apply(lambda x: x.autocorr(lag=1))
        features.append(vol_clustering.values.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _create_regime_transition_features(self, regimes: np.ndarray) -> np.ndarray:
        """Create regime-based transition features."""
        features = []
        regime_persistence = self._calculate_regime_persistence(regimes)
        features.append(regime_persistence.reshape(-1, 1))
        regime_stability = self._calculate_regime_stability(regimes)
        features.append(regime_stability.reshape(-1, 1))
        transition_prob = self._calculate_transition_probability(regimes)
        features.append(transition_prob.reshape(-1, 1))
        regime_duration = self._calculate_regime_duration(regimes)
        features.append(regime_duration.reshape(-1, 1))
        change_frequency = self._calculate_regime_change_frequency(regimes)
        features.append(change_frequency.reshape(-1, 1))
        regime_encoded = self._encode_regimes(regimes)
        features.append(regime_encoded)
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(regimes), 0)

    def _create_technical_transition_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create technical indicator transition features."""
        features = []
        rsi = self._calculate_rsi(data['close'])
        features.append(rsi.values.reshape(-1, 1))
        features.append(rsi.diff().values.reshape(-1, 1))
        macd = self._calculate_macd(data['close'])
        features.append(macd.values.reshape(-1, 1))
        features.append(macd.diff().values.reshape(-1, 1))
        bb_position, bb_width = self._calculate_bollinger_bands(data['close'])
        features.append(bb_position.values.reshape(-1, 1))
        features.append(bb_width.values.reshape(-1, 1))
        atr = self._calculate_atr(data)
        features.append(atr.values.reshape(-1, 1))
        features.append((atr / data['close']).values.reshape(-1, 1))
        adx = self._calculate_adx(data)
        features.append(adx.values.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _create_interaction_transition_features(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Create interaction features for transition prediction."""
        features = []
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        price_volume_interaction = price_change * volume_change
        features.append(price_volume_interaction.values.reshape(-1, 1))
        volatility = data['close'].pct_change().rolling(20).std()
        vol_vol_interaction = volatility * data['volume']
        features.append(vol_vol_interaction.values.reshape(-1, 1))
        momentum = data['close'].pct_change(10)
        regime_momentum_interaction = regimes.astype(float) * momentum
        features.append(regime_momentum_interaction.values.reshape(-1, 1))
        regime_vol_interaction = regimes.astype(float) * volatility
        features.append(regime_vol_interaction.values.reshape(-1, 1))
        price_position = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
        price_pos_regime_interaction = price_position * regimes.astype(float)
        features.append(price_pos_regime_interaction.values.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _create_lagged_features(self, data: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Create lagged features for transition prediction."""
        features = []
        for lag in [1, 2, 3, 5, 10]:
            lagged_price = data['close'].pct_change(lag)
            features.append(lagged_price.values.reshape(-1, 1))
            lagged_volume = data['volume'].pct_change(lag)
            features.append(lagged_volume.values.reshape(-1, 1))
            lagged_volatility = data['close'].pct_change().rolling(20).std().shift(lag)
            features.append(lagged_volatility.values.reshape(-1, 1))
        for lag in [1, 2, 3, 5]:
            lagged_regime = regimes.astype(float).shift(lag)
            features.append(lagged_regime.reshape(-1, 1))
        return np.concatenate(features, axis=1) if features else np.array([]).reshape(len(data), 0)

    def _handle_class_imbalance(self, labels: np.ndarray) -> np.ndarray:
        """Handle class imbalance in transition labels."""
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        self.class_weights = dict(zip(unique_classes, class_weights))
        return labels

    def _random_forest_feature_selection(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Use Random Forest for initial feature selection."""
        print('🔍 Running Random Forest feature selection...')
        rf_model = RandomForestClassifier(**self.rf_params)
        rf_model.fit(X, y)
        feature_importance = rf_model.feature_importances_
        perm_importance = permutation_importance(rf_model, X, y, n_repeats=10, random_state=self.random_state)
        perm_importance_mean = perm_importance.importances_mean
        combined_importance = 0.6 * feature_importance + 0.4 * perm_importance_mean
        feature_indices = np.argsort(combined_importance)[::-1]
        print(f'✅ Random Forest feature selection completed')
        print(f'   - Top 10 features by importance: {feature_indices[:10]}')
        print(f'   - Top 10 importance scores: {combined_importance[feature_indices[:10]]}')
        return {'feature_importance': feature_importance, 'permutation_importance': perm_importance_mean, 'combined_importance': combined_importance, 'feature_indices': feature_indices, 'rf_model': rf_model}

    def _lgbm_iterative_feature_selection(self, X: np.ndarray, y: np.ndarray, combined_importance: np.ndarray) -> Dict[str, Any]:
        """Use LGBM for iterative feature selection."""
        print('🔍 Running LGBM iterative feature selection...')
        feature_indices = np.argsort(combined_importance)[::-1]
        current_features = feature_indices[:self.initial_features]
        best_performance = 0.0
        best_features = current_features.copy()
        selection_history = []
        no_improvement_count = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        while len(current_features) < min(self.max_features, len(feature_indices)):
            X_train_current = X_train[:, current_features]
            X_test_current = X_test[:, current_features]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_current)
            X_test_scaled = scaler.transform(X_test_current)
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
            lgb_model = lgb.train(self.lgb_params, train_data, valid_sets=[val_data], num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            y_pred_proba = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
            performance = f1_score(y_test, y_pred)
            selection_history.append({'n_features': len(current_features), 'features': current_features.copy(), 'performance': performance, 'feature_names': [f'feature_{i}' for i in current_features]})
            print(f'   Features: {len(current_features)}, Performance: {performance:.4f}')
            if performance > best_performance + self.min_improvement:
                best_performance = performance
                best_features = current_features.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= self.patience:
                print(f'   Performance plateau detected. Testing feature removal...')
                if len(current_features) > 25:
                    feature_importance = lgb_model.feature_importance(importance_type='gain')
                    feature_importance_dict = dict(zip(current_features, feature_importance))
                    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1])
                    features_to_remove = [feat[0] for feat in sorted_features[:5]]
                    reduced_features = [f for f in current_features if f not in features_to_remove]
                    if len(reduced_features) >= 15:
                        X_train_reduced = X_train[:, reduced_features]
                        X_test_reduced = X_test[:, reduced_features]
                        scaler_reduced = StandardScaler()
                        X_train_scaled_reduced = scaler_reduced.fit_transform(X_train_reduced)
                        X_test_scaled_reduced = scaler_reduced.transform(X_test_reduced)
                        train_data_reduced = lgb.Dataset(X_train_scaled_reduced, label=y_train)
                        val_data_reduced = lgb.Dataset(X_test_scaled_reduced, label=y_test, reference=train_data_reduced)
                        lgb_model_reduced = lgb.train(self.lgb_params, train_data_reduced, valid_sets=[val_data_reduced], num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                        y_pred_proba_reduced = lgb_model_reduced.predict(X_test_scaled_reduced, num_iteration=lgb_model_reduced.best_iteration)
                        y_pred_reduced = (y_pred_proba_reduced > 0.5).astype(int)
                        performance_reduced = f1_score(y_test, y_pred_reduced)
                        print(f'   Reduced features: {len(reduced_features)}, Performance: {performance_reduced:.4f}')
                        recent_performances = [p['performance'] for p in selection_history[-3:]]
                        avg_recent_performance = np.mean(recent_performances)
                        if performance_reduced > avg_recent_performance:
                            print(f'   Feature removal improved performance. Using reduced feature set.')
                            current_features = reduced_features
                            best_performance = performance_reduced
                            best_features = reduced_features.copy()
                            no_improvement_count = 0
                            continue
                        else:
                            print(f'   Feature removal did not improve performance. Stopping.')
                            break
                    else:
                        print(f'   Not enough features to remove. Stopping.')
                        break
                else:
                    print(f'   Not enough features to remove. Stopping.')
                    break
            next_features = feature_indices[len(current_features):len(current_features) + self.feature_increment]
            if len(next_features) == 0:
                break
            current_features = np.concatenate([current_features, next_features])
        print(f'✅ LGBM iterative feature selection completed')
        print(f'   - Best performance: {best_performance:.4f}')
        print(f'   - Best features: {len(best_features)}')
        print(f'   - Feature indices: {best_features}')
        return {'selected_features': best_features, 'best_performance': best_performance, 'selection_history': selection_history}

    def _train_final_lgbm_model(self, X: np.ndarray, y: np.ndarray, selected_features: np.ndarray) -> Dict[str, Any]:
        """Train final LGBM model with selected features."""
        print('🔍 Training final LGBM model...')
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=self.random_state, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        final_model = lgb.train(self.lgb_params, train_data, valid_sets=[val_data], num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        y_pred_proba = final_model.predict(X_test_scaled, num_iteration=final_model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        performance = {'f1_score': f1_score(y_test, y_pred), 'roc_auc': roc_auc_score(y_test, y_pred_proba), 'accuracy': np.mean(y_test == y_pred)}
        print(f'✅ Final LGBM model trained')
        print(f"   - F1 Score: {performance['f1_score']:.4f}")
        print(f"   - ROC AUC: {performance['roc_auc']:.4f}")
        print(f"   - Accuracy: {performance['accuracy']:.4f}")
        self.best_lgb_model = final_model
        self.best_performance = performance['f1_score']
        self.selected_features = selected_features
        return {'model': final_model, 'scaler': scaler, 'performance': performance, 'selected_features': selected_features}

    def _generate_training_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training summary."""
        summary = {'feature_selection_completed': training_results.get('feature_selection_completed', False), 'lgb_training_completed': training_results.get('lgb_training_completed', False), 'n_selected_features': len(training_results.get('selected_features', [])), 'best_performance': training_results.get('best_performance', 0.0), 'final_performance': training_results.get('final_performance', {}), 'feature_selection_steps': len(training_results.get('feature_selection_history', []))}
        if training_results.get('selected_features'):
            selected_features = training_results['selected_features']
            summary['top_10_features'] = selected_features[:10].tolist()
        return summary

    def predict_transitions(self, data: pd.DataFrame, regimes: np.ndarray) -> Dict[str, Any]:
        """Predict regime transitions using trained models."""
        if self.best_lgb_model is None:
            return {'error': 'No trained model available'}
        try:
            features = self._create_comprehensive_transition_features(data, regimes)
            features_selected = features[:, self.selected_features]
            features_scaled = self.final_scaler.transform(features_selected)
            transition_probabilities = self.best_lgb_model.predict(features_scaled, num_iteration=self.best_lgb_model.best_iteration)
            transition_predictions = (transition_probabilities > 0.5).astype(int)
            return {'transition_predictions': transition_predictions, 'transition_probabilities': transition_probabilities, 'confidence_scores': transition_probabilities, 'model_used': 'lgbm_final', 'selected_features': self.selected_features}
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

    def save_models(self, filepath: str) -> bool:
        """Save trained models to file."""
        try:
            model_data = {'best_lgb_model': self.best_lgb_model, 'final_scaler': self.final_scaler, 'selected_features': self.selected_features, 'best_performance': self.best_performance, 'config': self.config}
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f'Failed to save models: {e}')
            return False

    def load_models(self, filepath: str) -> bool:
        """Load trained models from file."""
        try:
            model_data = joblib.load(filepath)
            self.best_lgb_model = model_data['best_lgb_model']
            self.final_scaler = model_data['final_scaler']
            self.selected_features = model_data['selected_features']
            self.best_performance = model_data['best_performance']
            self.config.update(model_data['config'])
            return True
        except Exception as e:
            print(f'Failed to load models: {e}')
            return False

    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime."""
        low_threshold = volatility.rolling(100).quantile(0.33)
        high_threshold = volatility.rolling(100).quantile(0.67)
        regime = pd.Series(1, index=volatility.index)
        regime[volatility > high_threshold] = 3
        regime[(volatility > low_threshold) & (volatility <= high_threshold)] = 2
        return regime.fillna(1)

    def _calculate_regime_persistence(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime persistence."""
        persistence = np.zeros(len(regimes))
        current_regime = regimes[0]
        current_count = 0
        for i in range(len(regimes)):
            if regimes[i] == current_regime:
                current_count += 1
            else:
                current_count = 1
                current_regime = regimes[i]
            persistence[i] = current_count
        return persistence

    def _calculate_regime_stability(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime stability."""
        stability = np.zeros(len(regimes))
        window = 20
        for i in range(len(regimes)):
            start_idx = max(0, i - window + 1)
            recent_regimes = regimes[start_idx:i + 1]
            stability[i] = 1 / (1 + np.std(recent_regimes))
        return stability

    def _calculate_transition_probability(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate transition probability."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        if n_regimes < 2:
            return np.zeros(len(regimes))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
        for i in range(len(regimes) - 1):
            current_idx = regime_map[regimes[i]]
            next_idx = regime_map[regimes[i + 1]]
            transition_matrix[current_idx, next_idx] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums > 0)
        transition_probs = np.zeros(len(regimes))
        for i in range(len(regimes)):
            current_idx = regime_map[regimes[i]]
            other_probs = transition_matrix[current_idx, :]
            other_probs[current_idx] = 0
            transition_probs[i] = np.sum(other_probs)
        return transition_probs

    def _calculate_regime_duration(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime duration."""
        duration = np.zeros(len(regimes))
        current_regime = regimes[0]
        current_duration = 0
        for i in range(len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = regimes[i]
            duration[i] = current_duration
        return duration

    def _calculate_regime_change_frequency(self, regimes: np.ndarray) -> np.ndarray:
        """Calculate regime change frequency."""
        change_freq = np.zeros(len(regimes))
        window = 50
        for i in range(len(regimes)):
            start_idx = max(0, i - window + 1)
            recent_regimes = regimes[start_idx:i + 1]
            changes = np.sum(np.diff(recent_regimes) != 0)
            change_freq[i] = changes / len(recent_regimes)
        return change_freq

    def _encode_regimes(self, regimes: np.ndarray) -> np.ndarray:
        """Encode regimes as one-hot vectors."""
        unique_regimes = np.unique(regimes)
        n_regimes = len(unique_regimes)
        if n_regimes <= 1:
            return np.zeros((len(regimes), 1))
        regime_encoded = np.zeros((len(regimes), n_regimes))
        for i, regime in enumerate(unique_regimes):
            regime_encoded[regimes == regime, i] = 1
        return regime_encoded

    def _calculate_rsi(self, prices: pd.Series, window: int=14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int=12, slow: int=26) -> pd.Series:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int=20, num_std: float=2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        bb_upper = sma + std * num_std
        bb_lower = sma - std * num_std
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        bb_width = (bb_upper - bb_lower) / sma
        return (bb_position, bb_width)

    def _calculate_atr(self, data: pd.DataFrame, window: int=14) -> pd.Series:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def _calculate_adx(self, data: pd.DataFrame, window: int=14) -> pd.Series:
        """Calculate ADX."""
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        return adx
if __name__ == '__main__':
    np.random.seed(42)
    n_samples = 2000
    regimes = np.zeros(n_samples, dtype=int)
    regimes[500:1000] = 1
    regimes[1000:1500] = 2
    regimes[1500:] = 1
    data = pd.DataFrame({'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.01), 'high': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) + np.abs(np.random.randn(n_samples) * 0.5), 'low': 100 + np.cumsum(np.random.randn(n_samples) * 0.01) - np.abs(np.random.randn(n_samples) * 0.5), 'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01), 'volume': np.random.lognormal(10, 1, n_samples)})
    config = {'initial_features': 20, 'feature_increment': 10, 'max_features': 100, 'min_improvement': 0.001, 'patience': 3, 'random_state': 42}
    detector = EnhancedMLRegimeTransitionDetector(config)
    training_results = detector.train_transition_models(data, regimes)
    print('Enhanced ML Transition Detection Training Results:')
    print(f"Feature selection completed: {training_results.get('feature_selection_completed', False)}")
    print(f"LGBM training completed: {training_results.get('lgb_training_completed', False)}")
    print(f"Selected features: {len(training_results.get('selected_features', []))}")
    print(f"Best performance: {training_results.get('best_performance', 0.0):.4f}")
    print(f"Final performance: {training_results.get('final_performance', {})}")
    predictions = detector.predict_transitions(data, regimes)
    print(f'\nPrediction Results:')
    print(f"Model used: {predictions.get('model_used', 'N/A')}")
    print(f"Transition predictions: {np.sum(predictions.get('transition_predictions', []))} transitions predicted")
    print(f"Mean confidence: {np.mean(predictions.get('confidence_scores', [0])):.4f}")
    detector.save_models('enhanced_transition_models.joblib')
    print('Models saved successfully')