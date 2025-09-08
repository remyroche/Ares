from ..standardized_parquet_handler import standardized_parquet_handler
"""
Unified Step08 Methods Implementation - Part 3
"""

    async def _advanced_feature_selection(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Advanced feature selection with bias prevention."""
        try:
            self.logger.info('🔍 Starting advanced feature selection...')
            
            # Extract features and target
            feature_columns = [col for col in data.columns if col not in ['composite_cluster_id', 'timestamp']]
            X = data[feature_columns]
            
            # Create target from regime labels for feature selection
            y = data['composite_cluster_id'].astype(int)
            
            # Phase 1: Initial feature selection
            self.logger.info('📊 Phase 1: Initial feature selection...')
            phase1_features = await self._phase1_feature_selection(X, y)
            
            # Phase 2: Advanced feature selection with bias prevention
            self.logger.info('🎯 Phase 2: Advanced feature selection with bias prevention...')
            phase2_features = await self._phase2_feature_selection(X[phase1_features], y)
            
            # Phase 3: Feature validation and stability assessment
            self.logger.info('✅ Phase 3: Feature validation and stability assessment...')
            validated_features = await self._validate_feature_stability(X, phase2_features, y)
            
            # Create final feature sets
            feature_sets = {
                'phase1': phase1_features,
                'phase2': phase2_features,
                'validated': validated_features,
                'final': validated_features  # Use validated features as final
            }
            
            # Add size-based feature sets
            for target_size in self.phase2_targets:
                if len(validated_features) >= target_size:
                    feature_sets[f'top_{target_size}'] = validated_features[:target_size]
            
            self.logger.info(f'✅ Feature selection completed: {len(feature_columns)} → {len(validated_features)} features')
            return feature_sets
            
        except Exception as e:
            self.logger.error(f'Failed to perform advanced feature selection: {e}')
            return {}

    async def _phase1_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Phase 1: Initial feature selection using mRMR and Random Forest."""
        try:
            feature_names = X.columns.tolist()
            X_values = X.values
            y_values = y.values
            
            # mRMR selection
            mrmr_features = []
            if self.enable_mrmr:
                self.logger.info('🔍 Running mRMR selection...')
                mrmr_features = self._mrmr_selection(X_values, y_values, feature_names, self.phase1_target_features)
                self.logger.info(f'   mRMR selected {len(mrmr_features)} features')
            
            # Random Forest selection
            rf_features = []
            if self.enable_rf_importance:
                self.logger.info('🌳 Running Random Forest selection...')
                rf_features = self._rf_selection(X_values, y_values, feature_names, self.phase1_target_features)
                self.logger.info(f'   RF selected {len(rf_features)} features')
            
            # Combine results
            consensus_features = list(set(mrmr_features) & set(rf_features))
            all_features = list(set(mrmr_features) | set(rf_features))
            
            # Select final features
            final_features = consensus_features.copy()
            remaining_slots = self.phase1_target_features - len(final_features)
            
            # Add remaining features from union
            for feature in all_features:
                if feature not in final_features and remaining_slots > 0:
                    final_features.append(feature)
                    remaining_slots -= 1
            
            self.logger.info(f'✅ Phase 1 complete: {len(feature_names)} → {len(final_features)} features')
            self.logger.info(f'   Consensus features: {len(consensus_features)}')
            
            return final_features
            
        except Exception as e:
            self.logger.error(f'Phase 1 feature selection failed: {e}')
            return []

    def _mrmr_selection(self, X_values: np.ndarray, y_values: np.ndarray, feature_names: List[str], n_features: int) -> List[str]:
        """Minimum Redundancy Maximum Relevance feature selection."""
        try:
            # Calculate relevance scores (mutual information)
            if NUMBA_AVAILABLE:
                relevance_scores = fast_mutual_info_discrete(X_values, y_values)
            else:
                relevance_scores = mutual_info_classif(X_values, y_values, random_state=42)
            
            # Calculate correlation matrix
            if NUMBA_AVAILABLE:
                corr_matrix = np.abs(fast_correlation_matrix(X_values))
            else:
                corr_matrix = np.abs(np.corrcoef(X_values.T))
            
            # mRMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(feature_names)))
            
            # Start with best feature
            first_idx = np.argmax(relevance_scores)
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select features
            while len(selected_indices) < n_features and remaining_indices:
                remaining_relevance = relevance_scores[remaining_indices]
                redundancy_scores = np.mean(corr_matrix[np.ix_(remaining_indices, selected_indices)], axis=1)
                mrmr_scores = remaining_relevance - redundancy_scores
                
                best_idx_in_remaining = np.argmax(mrmr_scores)
                best_idx = remaining_indices[best_idx_in_remaining]
                
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            return [feature_names[idx] for idx in selected_indices]
            
        except Exception as e:
            self.logger.error(f'mRMR selection failed: {e}')
            return []

    def _rf_selection(self, X_values: np.ndarray, y_values: np.ndarray, feature_names: List[str], n_features: int) -> List[str]:
        """Random Forest feature selection with time-series validation."""
        try:
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(5, 3))
            feature_importances = np.zeros(X_values.shape[1])
            
            for train_idx, val_idx in tscv.split(X_values):
                X_train, y_train = X_values[train_idx], y_values[train_idx]
                
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                feature_importances += rf.feature_importances_
            
            feature_importances /= tscv.get_n_splits()
            
            # Select top features
            top_indices = np.argsort(feature_importances)[-n_features:]
            return [feature_names[idx] for idx in top_indices]
            
        except Exception as e:
            self.logger.error(f'RF selection failed: {e}')
            return []

    async def _phase2_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Phase 2: Advanced feature selection with Boruta."""
        try:
            if not BORUTA_AVAILABLE:
                self.logger.warning('Boruta not available, using RF importance fallback')
                return self._rf_fallback_selection(X, y)
            
            self.logger.info('🔍 Running Boruta feature selection...')
            
            # Use Random Forest as base estimator
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize Boruta
            boruta = BorutaPy(
                rf,
                n_estimators='auto',
                alpha=self.boruta_alpha,
                max_iter=self.boruta_max_iter,
                random_state=42
            )
            
            # Fit Boruta
            boruta.fit(X.values, y.values)
            
            # Get results
            confirmed_features = X.columns[boruta.support_].tolist()
            tentative_features = X.columns[boruta.support_weak_].tolist()
            
            self.logger.info(f'   Boruta confirmed {len(confirmed_features)} features')
            self.logger.info(f'   Boruta tentative {len(tentative_features)} features')
            
            # Combine confirmed and tentative features
            final_features = confirmed_features + tentative_features
            
            return final_features
            
        except Exception as e:
            self.logger.error(f'Phase 2 feature selection failed: {e}')
            return X.columns.tolist()

    def _rf_fallback_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Fallback feature selection using Random Forest importance."""
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            
            # Select top features
            threshold = feature_importance.quantile(0.2)
            selected_features = feature_importance[feature_importance > threshold].index.tolist()
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f'RF fallback selection failed: {e}')
            return X.columns.tolist()

    async def _validate_feature_stability(self, X: pd.DataFrame, features: List[str], y: pd.Series) -> List[str]:
        """Validate feature stability across time and regimes."""
        try:
            self.logger.info('✅ Validating feature stability...')
            
            stable_features = []
            stability_scores = {}
            
            for feature in features:
                if feature not in X.columns:
                    continue
                
                # Calculate stability score
                stability_score = self._calculate_feature_stability(X[feature], y)
                stability_scores[feature] = stability_score
                
                # Keep features above stability threshold
                if stability_score >= self.feature_stability_threshold:
                    stable_features.append(feature)
            
            # Sort by stability score
            stable_features.sort(key=lambda x: stability_scores.get(x, 0), reverse=True)
            
            self.logger.info(f'   Feature stability validation: {len(features)} → {len(stable_features)} features')
            self.logger.info(f'   Average stability score: {np.mean(list(stability_scores.values())):.3f}')
            
            return stable_features
            
        except Exception as e:
            self.logger.error(f'Feature stability validation failed: {e}')
            return features

    def _calculate_feature_stability(self, feature_values: pd.Series, y: pd.Series) -> float:
        """Calculate feature stability score across time and regimes."""
        try:
            # Temporal stability (correlation with time)
            if len(feature_values) > 1:
                time_corr = np.abs(np.corrcoef(feature_values.values, range(len(feature_values)))[0, 1])
                temporal_stability = 1 - time_corr  # Lower correlation with time is better
            else:
                temporal_stability = 1.0
            
            # Regime stability (consistency across regimes)
            regime_stability = 0.0
            unique_regimes = y.unique()
            if len(unique_regimes) > 1:
                regime_means = []
                for regime in unique_regimes:
                    regime_data = feature_values[y == regime]
                    if len(regime_data) > 0:
                        regime_means.append(regime_data.mean())
                
                if len(regime_means) > 1:
                    regime_std = np.std(regime_means)
                    regime_mean = np.mean(regime_means)
                    if regime_mean != 0:
                        regime_stability = 1 - (regime_std / abs(regime_mean))
                    else:
                        regime_stability = 1 - regime_std
                    regime_stability = max(0, min(1, regime_stability))
                else:
                    regime_stability = 1.0
            else:
                regime_stability = 1.0
            
            # Overall stability score
            overall_stability = (temporal_stability + regime_stability) / 2
            
            return overall_stability
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate feature stability: {e}')
            return 0.5  # Default moderate stability

    async def _calculate_financial_metrics(self, data: pd.DataFrame, selected_features: Dict[str, List[str]]) -> FinancialMetrics:
        """Calculate comprehensive financial metrics."""
        try:
            self.logger.info('💰 Calculating financial metrics...')
            
            financial_metrics = FinancialMetrics()
            
            # Calculate returns
            if 'close' in data.columns:
                returns = self._calculate_returns(data['close'])
                financial_metrics.returns = {
                    'daily': returns.get('daily', 0.0),
                    'weekly': returns.get('weekly', 0.0),
                    'monthly': returns.get('monthly', 0.0),
                    'annualized': returns.get('annualized', 0.0)
                }
            
            # Calculate volatility
            if 'close' in data.columns:
                volatility = self._calculate_volatility(data['close'])
                financial_metrics.volatility = {
                    'daily': volatility.get('daily', 0.0),
                    'weekly': volatility.get('weekly', 0.0),
                    'monthly': volatility.get('monthly', 0.0),
                    'annualized': volatility.get('annualized', 0.0)
                }
            
            # Calculate Sharpe ratio
            if financial_metrics.returns and financial_metrics.volatility:
                sharpe_ratio = self._calculate_sharpe_ratio(
                    financial_metrics.returns['annualized'],
                    financial_metrics.volatility['annualized']
                )
                financial_metrics.sharpe_ratio = {
                    'overall': sharpe_ratio,
                    'regime_adjusted': self._calculate_regime_adjusted_sharpe(data, financial_metrics)
                }
            
            # Calculate VaR
            if 'close' in data.columns:
                var_metrics = self._calculate_var(data['close'])
                financial_metrics.var_95 = {'overall': var_metrics.get('var_95', 0.0)}
                financial_metrics.var_99 = {'overall': var_metrics.get('var_99', 0.0)}
            
            # Calculate maximum drawdown
            if 'close' in data.columns:
                max_dd = self._calculate_max_drawdown(data['close'])
                financial_metrics.max_drawdown = {'overall': max_dd}
            
            # Calculate Calmar ratio
            if financial_metrics.returns and financial_metrics.max_drawdown:
                calmar_ratio = financial_metrics.returns['annualized'] / abs(financial_metrics.max_drawdown['overall'])
                financial_metrics.calmar_ratio = {'overall': calmar_ratio}
            
            # Calculate Sortino ratio
            if 'close' in data.columns:
                sortino_ratio = self._calculate_sortino_ratio(data['close'])
                financial_metrics.sortino_ratio = {'overall': sortino_ratio}
            
            # Calculate regime-specific metrics
            regime_metrics = self._calculate_regime_specific_metrics(data)
            for metric_name, metric_values in regime_metrics.items():
                if hasattr(financial_metrics, metric_name):
                    setattr(financial_metrics, metric_name, metric_values)
            
            self.logger.info('✅ Financial metrics calculated successfully')
            return financial_metrics
            
        except Exception as e:
            self.logger.error(f'Failed to calculate financial metrics: {e}')
            return FinancialMetrics()

    def _calculate_returns(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various return metrics."""
        try:
            # Daily returns
            daily_returns = prices.pct_change().dropna()
            
            # Calculate different period returns
            returns = {
                'daily': daily_returns.mean(),
                'weekly': daily_returns.resample('W').apply(lambda x: (1 + x).prod() - 1).mean(),
                'monthly': daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1).mean(),
                'annualized': daily_returns.mean() * 252  # Assuming 252 trading days
            }
            
            return returns
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate returns: {e}')
            return {'daily': 0.0, 'weekly': 0.0, 'monthly': 0.0, 'annualized': 0.0}

    def _calculate_volatility(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various volatility metrics."""
        try:
            # Daily returns
            daily_returns = prices.pct_change().dropna()
            
            # Calculate different period volatilities
            volatility = {
                'daily': daily_returns.std(),
                'weekly': daily_returns.resample('W').std().mean(),
                'monthly': daily_returns.resample('M').std().mean(),
                'annualized': daily_returns.std() * np.sqrt(252)  # Assuming 252 trading days
            }
            
            return volatility
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate volatility: {e}')
            return {'daily': 0.0, 'weekly': 0.0, 'monthly': 0.0, 'annualized': 0.0}

    def _calculate_sharpe_ratio(self, annual_return: float, annual_volatility: float) -> float:
        """Calculate Sharpe ratio."""
        try:
            if annual_volatility == 0:
                return 0.0
            
            excess_return = annual_return - self.risk_free_rate
            sharpe_ratio = excess_return / annual_volatility
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate Sharpe ratio: {e}')
            return 0.0

    def _calculate_regime_adjusted_sharpe(self, data: pd.DataFrame, financial_metrics: FinancialMetrics) -> float:
        """Calculate regime-adjusted Sharpe ratio."""
        try:
            if 'composite_cluster_id' not in data.columns or 'close' not in data.columns:
                return financial_metrics.sharpe_ratio.get('overall', 0.0)
            
            regime_sharpes = []
            for regime in data['composite_cluster_id'].unique():
                regime_data = data[data['composite_cluster_id'] == regime]
                if len(regime_data) < 10:  # Need minimum samples
                    continue
                
                regime_returns = regime_data['close'].pct_change().dropna()
                if len(regime_returns) == 0:
                    continue
                
                regime_return = regime_returns.mean() * 252
                regime_volatility = regime_returns.std() * np.sqrt(252)
                
                if regime_volatility > 0:
                    regime_sharpe = (regime_return - self.risk_free_rate) / regime_volatility
                    regime_sharpes.append(regime_sharpe)
            
            if regime_sharpes:
                # Weight by regime frequency
                regime_weights = []
                for regime in data['composite_cluster_id'].unique():
                    weight = (data['composite_cluster_id'] == regime).sum() / len(data)
                    regime_weights.append(weight)
                
                regime_weights = np.array(regime_weights[:len(regime_sharpes)])
                regime_weights = regime_weights / regime_weights.sum()
                
                weighted_sharpe = np.average(regime_sharpes, weights=regime_weights)
                return weighted_sharpe
            else:
                return financial_metrics.sharpe_ratio.get('overall', 0.0)
                
        except Exception as e:
            self.logger.warning(f'Failed to calculate regime-adjusted Sharpe ratio: {e}')
            return financial_metrics.sharpe_ratio.get('overall', 0.0)

    def _calculate_var(self, prices: pd.Series, confidence_levels: List[float] = None) -> Dict[str, float]:
        """Calculate Value at Risk (VaR)."""
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]
            
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return {'var_95': 0.0, 'var_99': 0.0}
            
            var_metrics = {}
            for conf_level in confidence_levels:
                var_value = np.percentile(returns, (1 - conf_level) * 100)
                var_metrics[f'var_{int(conf_level * 100)}'] = var_value
            
            return var_metrics
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate VaR: {e}')
            return {'var_95': 0.0, 'var_99': 0.0}

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + prices.pct_change()).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            
            return max_drawdown
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate maximum drawdown: {e}')
            return 0.0

    def _calculate_sortino_ratio(self, prices: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            # Calculate downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float('inf')  # No negative returns
            
            downside_deviation = negative_returns.std() * np.sqrt(252)
            
            # Calculate Sortino ratio
            annual_return = returns.mean() * 252
            excess_return = annual_return - self.risk_free_rate
            
            if downside_deviation == 0:
                return 0.0
            
            sortino_ratio = excess_return / downside_deviation
            
            return sortino_ratio
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate Sortino ratio: {e}')
            return 0.0

    def _calculate_regime_specific_metrics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate regime-specific financial metrics."""
        try:
            if 'composite_cluster_id' not in data.columns or 'close' not in data.columns:
                return {}
            
            regime_metrics = {}
            
            for regime in data['composite_cluster_id'].unique():
                regime_data = data[data['composite_cluster_id'] == regime]
                if len(regime_data) < 10:  # Need minimum samples
                    continue
                
                regime_returns = regime_data['close'].pct_change().dropna()
                if len(regime_returns) == 0:
                    continue
                
                regime_return = regime_returns.mean() * 252
                regime_volatility = regime_returns.std() * np.sqrt(252)
                
                if regime_volatility > 0:
                    regime_sharpe = (regime_return - self.risk_free_rate) / regime_volatility
                else:
                    regime_sharpe = 0.0
                
                regime_metrics[f'regime_{regime}'] = {
                    'return': regime_return,
                    'volatility': regime_volatility,
                    'sharpe_ratio': regime_sharpe
                }
            
            return regime_metrics
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate regime-specific metrics: {e}')
            return {}