from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Unified Step08 Risk Assessment and Validation Methods - Part 4
"""

    async def _comprehensive_risk_assessment(self, data: pd.DataFrame, selected_features: Dict[str, List[str]], financial_metrics: FinancialMetrics) -> RiskMetrics:
        """Comprehensive risk assessment with explicit risk metrics."""
        try:
            self.logger.info('⚠️ Performing comprehensive risk assessment...')
            
            risk_metrics = RiskMetrics()
            
            # Portfolio VaR calculation
            if 'close' in data.columns:
                portfolio_var = self._calculate_portfolio_var(data['close'])
                risk_metrics.portfolio_var = portfolio_var
            
            # Portfolio Expected Shortfall (ES)
            if 'close' in data.columns:
                portfolio_es = self._calculate_expected_shortfall(data['close'])
                risk_metrics.portfolio_es = portfolio_es
            
            # Concentration risk
            concentration_risk = self._calculate_concentration_risk(selected_features)
            risk_metrics.concentration_risk = concentration_risk
            
            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(data)
            risk_metrics.liquidity_risk = liquidity_risk
            
            # Model risk
            model_risk = self._calculate_model_risk(selected_features, data)
            risk_metrics.model_risk = model_risk
            
            # Regime risk
            regime_risk = self._calculate_regime_risk(data)
            risk_metrics.regime_risk = regime_risk
            
            # Feature stability risk
            feature_stability_risk = self._calculate_feature_stability_risk(selected_features, data)
            risk_metrics.feature_stability_risk = feature_stability_risk
            
            # Overfitting risk
            overfitting_risk = self._calculate_overfitting_risk(selected_features, data)
            risk_metrics.overfitting_risk = overfitting_risk
            
            # Data quality risk
            data_quality_risk = self._calculate_data_quality_risk(data)
            risk_metrics.data_quality_risk = data_quality_risk
            
            # Operational risk
            operational_risk = self._calculate_operational_risk()
            risk_metrics.operational_risk = operational_risk
            
            # Overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_metrics)
            risk_metrics.overall_risk_score = overall_risk_score
            
            self.logger.info(f'✅ Risk assessment completed:')
            self.logger.info(f'   Portfolio VaR: {portfolio_var:.4f}')
            self.logger.info(f'   Portfolio ES: {portfolio_es:.4f}')
            self.logger.info(f'   Model Risk: {model_risk:.4f}')
            self.logger.info(f'   Regime Risk: {regime_risk:.4f}')
            self.logger.info(f'   Overall Risk Score: {overall_risk_score:.4f}')
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f'Failed to perform risk assessment: {e}')
            return RiskMetrics()

    def _calculate_portfolio_var(self, prices: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            var_value = np.percentile(returns, (1 - confidence_level) * 100)
            return var_value
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate portfolio VaR: {e}')
            return 0.0

    def _calculate_expected_shortfall(self, prices: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            
            var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) == 0:
                return var_threshold
            
            expected_shortfall = tail_returns.mean()
            return expected_shortfall
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate Expected Shortfall: {e}')
            return 0.0

    def _calculate_concentration_risk(self, selected_features: Dict[str, List[str]]) -> float:
        """Calculate concentration risk based on feature selection."""
        try:
            if not selected_features or 'final' not in selected_features:
                return 0.0
            
            final_features = selected_features['final']
            if not final_features:
                return 0.0
            
            # Calculate feature concept concentration
            concept_counts = {}
            for feature in final_features:
                concept = self._identify_feature_concept(feature)
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            total_features = len(final_features)
            hhi = sum((count / total_features) ** 2 for count in concept_counts.values())
            
            # Convert to concentration risk (0-1, higher is more concentrated)
            concentration_risk = min(1.0, hhi)
            
            return concentration_risk
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate concentration risk: {e}')
            return 0.0

    def _identify_feature_concept(self, feature_name: str) -> str:
        """Identify the concept/category of a feature."""
        feature_lower = feature_name.lower()
        
        concept_patterns = {
            'momentum': ['rsi', 'macd', 'momentum', 'roc', 'stoch'],
            'volatility': ['bb_', 'atr', 'volatility', 'std', 'bollinger'],
            'volume': ['volume', 'vwap', 'obv', 'mfi', 'ad'],
            'trend': ['ema', 'sma', 'trend', 'adx', 'dmi'],
            'microstructure': ['spread', 'imbalance', 'flow', 'tick', 'bid', 'ask'],
            'regime': ['regime', 'cluster', 'state', 'hmm'],
            'support_resistance': ['sr_', 'support', 'resistance', 'level', 'pivot']
        }
        
        for concept, patterns in concept_patterns.items():
            if any(pattern in feature_lower for pattern in patterns):
                return concept
        
        return 'other'

    def _calculate_liquidity_risk(self, data: pd.DataFrame) -> float:
        """Calculate liquidity risk based on volume and price impact."""
        try:
            if 'volume' not in data.columns or 'close' not in data.columns:
                return 0.0
            
            # Calculate volume volatility
            volume_returns = data['volume'].pct_change().dropna()
            volume_volatility = volume_returns.std()
            
            # Calculate price impact (correlation between volume and price changes)
            price_returns = data['close'].pct_change().dropna()
            if len(volume_returns) > 0 and len(price_returns) > 0:
                min_len = min(len(volume_returns), len(price_returns))
                volume_returns = volume_returns.iloc[:min_len]
                price_returns = price_returns.iloc[:min_len]
                
                price_impact = abs(np.corrcoef(volume_returns, price_returns)[0, 1])
            else:
                price_impact = 0.0
            
            # Combine volume volatility and price impact
            liquidity_risk = (volume_volatility + price_impact) / 2
            
            return min(1.0, liquidity_risk)
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate liquidity risk: {e}')
            return 0.0

    def _calculate_model_risk(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Calculate model risk based on feature selection and data characteristics."""
        try:
            model_risk_factors = []
            
            # Feature selection complexity risk
            if 'final' in selected_features:
                n_features = len(selected_features['final'])
                n_samples = len(data)
                feature_ratio = n_features / n_samples
                
                # Higher feature-to-sample ratio increases model risk
                complexity_risk = min(1.0, feature_ratio * 10)  # Scale factor
                model_risk_factors.append(complexity_risk)
            
            # Data quality risk
            if 'close' in data.columns:
                price_volatility = data['close'].pct_change().std()
                # Higher volatility increases model risk
                volatility_risk = min(1.0, price_volatility * 100)  # Scale factor
                model_risk_factors.append(volatility_risk)
            
            # Regime stability risk
            if 'composite_cluster_id' in data.columns:
                regime_changes = (data['composite_cluster_id'].diff() != 0).sum()
                regime_stability_risk = min(1.0, regime_changes / len(data))
                model_risk_factors.append(regime_stability_risk)
            
            # Calculate overall model risk
            if model_risk_factors:
                model_risk = np.mean(model_risk_factors)
            else:
                model_risk = 0.0
            
            return model_risk
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate model risk: {e}')
            return 0.0

    def _calculate_regime_risk(self, data: pd.DataFrame) -> float:
        """Calculate regime-specific risk."""
        try:
            if 'composite_cluster_id' not in data.columns:
                return 0.0
            
            regime_risks = []
            
            for regime in data['composite_cluster_id'].unique():
                regime_data = data[data['composite_cluster_id'] == regime]
                if len(regime_data) < 10:  # Need minimum samples
                    continue
                
                # Calculate regime-specific volatility
                if 'close' in regime_data.columns:
                    regime_returns = regime_data['close'].pct_change().dropna()
                    if len(regime_returns) > 0:
                        regime_volatility = regime_returns.std()
                        regime_risks.append(regime_volatility)
            
            if regime_risks:
                # Calculate regime risk as weighted average of regime volatilities
                regime_weights = []
                for regime in data['composite_cluster_id'].unique():
                    weight = (data['composite_cluster_id'] == regime).sum() / len(data)
                    regime_weights.append(weight)
                
                regime_weights = np.array(regime_weights[:len(regime_risks)])
                regime_weights = regime_weights / regime_weights.sum()
                
                weighted_regime_risk = np.average(regime_risks, weights=regime_weights)
                return min(1.0, weighted_regime_risk * 100)  # Scale factor
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Failed to calculate regime risk: {e}')
            return 0.0

    def _calculate_feature_stability_risk(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Calculate feature stability risk."""
        try:
            if 'final' not in selected_features or not selected_features['final']:
                return 0.0
            
            final_features = selected_features['final']
            stability_scores = []
            
            for feature in final_features:
                if feature in data.columns:
                    stability_score = self._calculate_feature_stability(data[feature], data.get('composite_cluster_id', pd.Series()))
                    stability_scores.append(stability_score)
            
            if stability_scores:
                # Feature stability risk is inverse of average stability
                avg_stability = np.mean(stability_scores)
                stability_risk = 1 - avg_stability
                return stability_risk
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Failed to calculate feature stability risk: {e}')
            return 0.0

    def _calculate_overfitting_risk(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Calculate overfitting risk."""
        try:
            overfitting_factors = []
            
            # Feature-to-sample ratio
            if 'final' in selected_features:
                n_features = len(selected_features['final'])
                n_samples = len(data)
                if n_samples > 0:
                    feature_ratio = n_features / n_samples
                    # Higher ratio increases overfitting risk
                    overfitting_factors.append(min(1.0, feature_ratio * 5))
            
            # Feature selection complexity
            if 'phase1' in selected_features and 'phase2' in selected_features:
                phase1_count = len(selected_features['phase1'])
                phase2_count = len(selected_features['phase2'])
                if phase1_count > 0:
                    selection_ratio = phase2_count / phase1_count
                    # Lower ratio (more aggressive selection) increases overfitting risk
                    overfitting_factors.append(1 - selection_ratio)
            
            # Data quality impact
            if 'close' in data.columns:
                price_volatility = data['close'].pct_change().std()
                # Higher volatility increases overfitting risk
                overfitting_factors.append(min(1.0, price_volatility * 50))
            
            if overfitting_factors:
                overfitting_risk = np.mean(overfitting_factors)
            else:
                overfitting_risk = 0.0
            
            return overfitting_risk
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate overfitting risk: {e}')
            return 0.0

    def _calculate_data_quality_risk(self, data: pd.DataFrame) -> float:
        """Calculate data quality risk."""
        try:
            quality_factors = []
            
            # Missing data risk
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            quality_factors.append(missing_ratio)
            
            # Duplicate data risk
            if 'timestamp' in data.columns:
                duplicate_ratio = data['timestamp'].duplicated().sum() / len(data)
                quality_factors.append(duplicate_ratio)
            
            # Outlier risk (for numeric columns)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                outlier_ratios = []
                for col in numeric_columns:
                    if col in data.columns:
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                            outlier_ratio = outliers / len(data)
                            outlier_ratios.append(outlier_ratio)
                
                if outlier_ratios:
                    avg_outlier_ratio = np.mean(outlier_ratios)
                    quality_factors.append(avg_outlier_ratio)
            
            if quality_factors:
                data_quality_risk = np.mean(quality_factors)
            else:
                data_quality_risk = 0.0
            
            return min(1.0, data_quality_risk)
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate data quality risk: {e}')
            return 0.0

    def _calculate_operational_risk(self) -> float:
        """Calculate operational risk based on system configuration."""
        try:
            operational_factors = []
            
            # Dependency availability risk
            dependency_risk = 0.0
            if not ENHANCED_OPTIMIZATIONS_AVAILABLE:
                dependency_risk += 0.2
            if not BORUTA_AVAILABLE:
                dependency_risk += 0.1
            if not SHAP_AVAILABLE:
                dependency_risk += 0.1
            if not LIME_AVAILABLE:
                dependency_risk += 0.1
            
            operational_factors.append(dependency_risk)
            
            # Configuration complexity risk
            config_complexity = len(self.step_config) / 50  # Normalize by expected config size
            operational_factors.append(min(1.0, config_complexity))
            
            # Resource constraints risk
            resource_risk = 0.0
            if self.phase1_target_features > 200:
                resource_risk += 0.2
            if len(self.phase2_targets) > 5:
                resource_risk += 0.1
            
            operational_factors.append(resource_risk)
            
            if operational_factors:
                operational_risk = np.mean(operational_factors)
            else:
                operational_risk = 0.0
            
            return operational_risk
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate operational risk: {e}')
            return 0.0

    def _calculate_overall_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """Calculate overall risk score from individual risk components."""
        try:
            risk_components = [
                risk_metrics.portfolio_var,
                risk_metrics.portfolio_es,
                risk_metrics.concentration_risk,
                risk_metrics.liquidity_risk,
                risk_metrics.model_risk,
                risk_metrics.regime_risk,
                risk_metrics.feature_stability_risk,
                risk_metrics.overfitting_risk,
                risk_metrics.data_quality_risk,
                risk_metrics.operational_risk
            ]
            
            # Remove None values and calculate weighted average
            valid_risks = [r for r in risk_components if r is not None]
            
            if valid_risks:
                # Weight financial risks more heavily
                weights = [0.15, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]
                valid_weights = weights[:len(valid_risks)]
                valid_weights = np.array(valid_weights) / np.sum(valid_weights)
                
                overall_risk = np.average(valid_risks, weights=valid_weights)
            else:
                overall_risk = 0.0
            
            return min(1.0, overall_risk)
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate overall risk score: {e}')
            return 0.0

    async def _validate_feature_selection(self, data: pd.DataFrame, selected_features: Dict[str, List[str]]) -> FeatureSelectionValidation:
        """Validate feature selection to prevent bias."""
        try:
            self.logger.info('✅ Validating feature selection...')
            
            validation = FeatureSelectionValidation()
            
            # Selection bias assessment
            selection_bias_score = self._assess_selection_bias(selected_features, data)
            validation.selection_bias_score = selection_bias_score
            
            # Temporal stability validation
            temporal_stability = self._validate_temporal_stability(selected_features, data)
            validation.temporal_stability = temporal_stability
            
            # Regime consistency validation
            regime_consistency = self._validate_regime_consistency(selected_features, data)
            validation.regime_consistency = regime_consistency
            
            # Correlation stability validation
            correlation_stability = self._validate_correlation_stability(selected_features, data)
            validation.correlation_stability = correlation_stability
            
            # Importance stability validation
            importance_stability = self._validate_importance_stability(selected_features, data)
            validation.importance_stability = importance_stability
            
            # Overfitting indicators
            overfitting_indicators = self._assess_overfitting_indicators(selected_features, data)
            validation.overfitting_indicators = overfitting_indicators
            
            # Overall validation
            validation_scores = [
                selection_bias_score,
                temporal_stability,
                regime_consistency,
                correlation_stability,
                importance_stability
            ]
            
            avg_validation_score = np.mean(validation_scores)
            validation.validation_passed = avg_validation_score >= 0.7
            
            # Generate warnings
            if selection_bias_score < 0.5:
                validation.warnings.append("High selection bias detected")
            if temporal_stability < 0.6:
                validation.warnings.append("Low temporal stability")
            if regime_consistency < 0.6:
                validation.warnings.append("Low regime consistency")
            if correlation_stability < 0.6:
                validation.warnings.append("Low correlation stability")
            if importance_stability < 0.6:
                validation.warnings.append("Low importance stability")
            
            self.logger.info(f'✅ Feature selection validation completed:')
            self.logger.info(f'   Selection bias score: {selection_bias_score:.3f}')
            self.logger.info(f'   Temporal stability: {temporal_stability:.3f}')
            self.logger.info(f'   Regime consistency: {regime_consistency:.3f}')
            self.logger.info(f'   Validation passed: {validation.validation_passed}')
            
            return validation
            
        except Exception as e:
            self.logger.error(f'Failed to validate feature selection: {e}')
            return FeatureSelectionValidation()

    def _assess_selection_bias(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Assess selection bias in feature selection process."""
        try:
            if 'final' not in selected_features or not selected_features['final']:
                return 0.0
            
            final_features = selected_features['final']
            
            # Check for concept bias
            concept_counts = {}
            for feature in final_features:
                concept = self._identify_feature_concept(feature)
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Calculate concept diversity (inverse of concentration)
            total_features = len(final_features)
            if total_features == 0:
                return 0.0
            
            concept_diversity = len(concept_counts) / 7  # 7 main concepts
            concept_balance = 1 - (max(concept_counts.values()) / total_features)
            
            # Check for temporal bias (features from specific time periods)
            temporal_bias = 0.0
            if 'timestamp' in data.columns:
                # This is a simplified check - in practice, you'd analyze feature creation timestamps
                temporal_bias = 0.1  # Placeholder
            
            # Overall bias score
            bias_score = (concept_diversity + concept_balance + (1 - temporal_bias)) / 3
            
            return min(1.0, bias_score)
            
        except Exception as e:
            self.logger.warning(f'Failed to assess selection bias: {e}')
            return 0.0

    def _validate_temporal_stability(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Validate temporal stability of selected features."""
        try:
            if 'final' not in selected_features or not selected_features['final']:
                return 0.0
            
            final_features = selected_features['final']
            stability_scores = []
            
            for feature in final_features:
                if feature in data.columns:
                    # Calculate feature stability over time
                    feature_values = data[feature]
                    if len(feature_values) > 1:
                        # Calculate rolling correlation with time
                        time_index = np.arange(len(feature_values))
                        correlation = np.abs(np.corrcoef(feature_values, time_index)[0, 1])
                        stability = 1 - correlation  # Lower correlation with time is better
                        stability_scores.append(stability)
            
            if stability_scores:
                return np.mean(stability_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Failed to validate temporal stability: {e}')
            return 0.0

    def _validate_regime_consistency(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Validate regime consistency of selected features."""
        try:
            if 'final' not in selected_features or not selected_features['final'] or 'composite_cluster_id' not in data.columns:
                return 0.0
            
            final_features = selected_features['final']
            consistency_scores = []
            
            for feature in final_features:
                if feature in data.columns:
                    # Calculate feature consistency across regimes
                    regime_means = []
                    for regime in data['composite_cluster_id'].unique():
                        regime_data = data[data['composite_cluster_id'] == regime]
                        if len(regime_data) > 0 and feature in regime_data.columns:
                            regime_mean = regime_data[feature].mean()
                            regime_means.append(regime_mean)
                    
                    if len(regime_means) > 1:
                        # Calculate coefficient of variation
                        mean_val = np.mean(regime_means)
                        std_val = np.std(regime_means)
                        if mean_val != 0:
                            cv = std_val / abs(mean_val)
                            consistency = 1 - min(1.0, cv)  # Lower CV is better
                            consistency_scores.append(consistency)
            
            if consistency_scores:
                return np.mean(consistency_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Failed to validate regime consistency: {e}')
            return 0.0

    def _validate_correlation_stability(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Validate correlation stability of selected features."""
        try:
            if 'final' not in selected_features or not selected_features['final']:
                return 0.0
            
            final_features = selected_features['final']
            available_features = [f for f in final_features if f in data.columns]
            
            if len(available_features) < 2:
                return 0.0
            
            # Calculate correlation matrix
            feature_data = data[available_features]
            corr_matrix = feature_data.corr()
            
            # Check for high correlations (potential redundancy)
            high_corr_pairs = 0
            total_pairs = 0
            
            for i in range(len(available_features)):
                for j in range(i + 1, len(available_features)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    total_pairs += 1
                    if corr_value > 0.8:  # High correlation threshold
                        high_corr_pairs += 1
            
            if total_pairs > 0:
                redundancy_ratio = high_corr_pairs / total_pairs
                stability = 1 - redundancy_ratio
            else:
                stability = 1.0
            
            return stability
            
        except Exception as e:
            self.logger.warning(f'Failed to validate correlation stability: {e}')
            return 0.0

    def _validate_importance_stability(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> float:
        """Validate importance stability of selected features."""
        try:
            if 'final' not in selected_features or not selected_features['final']:
                return 0.0
            
            final_features = selected_features['final']
            available_features = [f for f in final_features if f in data.columns]
            
            if len(available_features) < 2:
                return 0.0
            
            # Calculate feature importance using Random Forest
            X = data[available_features]
            y = data.get('composite_cluster_id', pd.Series())
            
            if len(y) == 0 or len(y.unique()) < 2:
                return 0.0
            
            # Use time series split for stability assessment
            tscv = TimeSeriesSplit(n_splits=3)
            importance_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X_train, y_train)
                
                importance_scores.append(rf.feature_importances_)
            
            if len(importance_scores) > 1:
                # Calculate stability as inverse of variance in importance scores
                importance_array = np.array(importance_scores)
                importance_std = np.std(importance_array, axis=0)
                importance_mean = np.mean(importance_array, axis=0)
                
                # Avoid division by zero
                stability_scores = []
                for i in range(len(available_features)):
                    if importance_mean[i] > 0:
                        stability = 1 - (importance_std[i] / importance_mean[i])
                        stability_scores.append(max(0, stability))
                    else:
                        stability_scores.append(0)
                
                return np.mean(stability_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f'Failed to validate importance stability: {e}')
            return 0.0

    def _assess_overfitting_indicators(self, selected_features: Dict[str, List[str]], data: pd.DataFrame) -> Dict[str, float]:
        """Assess overfitting indicators."""
        try:
            indicators = {}
            
            # Feature-to-sample ratio
            if 'final' in selected_features:
                n_features = len(selected_features['final'])
                n_samples = len(data)
                if n_samples > 0:
                    indicators['feature_sample_ratio'] = n_features / n_samples
            
            # Selection aggressiveness
            if 'phase1' in selected_features and 'phase2' in selected_features:
                phase1_count = len(selected_features['phase1'])
                phase2_count = len(selected_features['phase2'])
                if phase1_count > 0:
                    indicators['selection_aggressiveness'] = 1 - (phase2_count / phase1_count)
            
            # Feature complexity (number of unique concepts)
            if 'final' in selected_features:
                concepts = set()
                for feature in selected_features['final']:
                    concepts.add(self._identify_feature_concept(feature))
                indicators['concept_diversity'] = len(concepts) / 7  # 7 main concepts
            
            return indicators
            
        except Exception as e:
            self.logger.warning(f'Failed to assess overfitting indicators: {e}')
            return {}