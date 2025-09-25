"""
Extended functionality for OptimizedTrainer - Hyperparameter Optimization and Advanced Features
"""

# Add these methods to the OptimizedTrainer class

def hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Dict[str, Any],
                               method: str = "grid",
                               cv_folds: int = 5,
                               n_trials: int = 50,
                               timeout: int = 3600) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization with M1 acceleration.
    
    Args:
        X: Features
        y: Target variable
        param_grid: Parameter grid for optimization
        method: Optimization method ('grid', 'random', 'bayesian')
        cv_folds: Number of CV folds
        n_trials: Number of trials for random/bayesian search
        timeout: Timeout in seconds
        
    Returns:
        Optimization results
    """
    _safe_print(f"🔍 Starting hyperparameter optimization using {method} search...")
    
    start_time = time.time()
    
    try:
        if method == "grid" and SKLEARN_AVAILABLE:
            return self._grid_search_optimization(X, y, param_grid, cv_folds)
        elif method == "random" and SKLEARN_AVAILABLE:
            return self._random_search_optimization(X, y, param_grid, cv_folds, n_trials)
        elif method == "bayesian" and OPTUNA_AVAILABLE:
            return self._bayesian_optimization(X, y, param_grid, n_trials, timeout)
        else:
            raise ValueError(f"Optimization method '{method}' not available or not supported")
            
    except Exception as e:
        self.logger.error(f"Hyperparameter optimization failed: {e}")
        raise

def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray,
                             param_grid: Dict[str, Any],
                             cv_folds: int) -> Dict[str, Any]:
    """Grid search optimization."""
    _safe_print("🔍 Performing Grid Search optimization...")
    
    try:
        # Create base estimator (placeholder - would be actual model)
        from sklearn.ensemble import RandomForestClassifier
        base_estimator = RandomForestClassifier(random_state=42)
        
        # Setup GridSearchCV with M1 optimization
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=cv_folds,
            n_jobs=self.cpu_optimizer.get_optimal_worker_count() if self.cpu_optimizer else -1,
            scoring='accuracy',
            verbose=1
        )
        
        # Perform search
        grid_search.fit(X, y)
        
        results = {
            'method': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'n_trials': len(grid_search.cv_results_['params'])
        }
        
        _safe_print(f"✅ Grid Search completed: Best score = {grid_search.best_score_:.4f}")
        return results
        
    except Exception as e:
        self.logger.error(f"Grid search failed: {e}")
        raise

def _random_search_optimization(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Dict[str, Any],
                               cv_folds: int, n_trials: int) -> Dict[str, Any]:
    """Random search optimization."""
    _safe_print(f"🎲 Performing Random Search optimization ({n_trials} trials)...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        base_estimator = RandomForestClassifier(random_state=42)
        
        # Setup RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=cv_folds,
            n_jobs=self.cpu_optimizer.get_optimal_worker_count() if self.cpu_optimizer else -1,
            scoring='accuracy',
            random_state=42,
            verbose=1
        )
        
        # Perform search
        random_search.fit(X, y)
        
        results = {
            'method': 'random_search',
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_,
            'n_trials': n_trials
        }
        
        _safe_print(f"✅ Random Search completed: Best score = {random_search.best_score_:.4f}")
        return results
        
    except Exception as e:
        self.logger.error(f"Random search failed: {e}")
        raise

def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray,
                          param_grid: Dict[str, Any],
                          n_trials: int, timeout: int) -> Dict[str, Any]:
    """Bayesian optimization using Optuna."""
    _safe_print(f"🧠 Performing Bayesian optimization ({n_trials} trials)...")
    
    try:
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Range specification
                    if all(isinstance(v, int) for v in param_values):
                        params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
            
            # Create and train model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            model = RandomForestClassifier(**params, random_state=42)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            return cv_scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        results = {
            'method': 'bayesian_optimization',
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
        
        _safe_print(f"✅ Bayesian optimization completed: Best score = {study.best_value:.4f}")
        return results
        
    except Exception as e:
        self.logger.error(f"Bayesian optimization failed: {e}")
        raise

def cross_validate(self, X: np.ndarray, y: np.ndarray,
                  cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
    """
    Perform cross-validation with M1 optimization.
    
    Args:
        X: Features
        y: Target variable
        cv_folds: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        CV results
    """
    _safe_print(f"🔄 Performing {cv_folds}-fold cross-validation...")
    
    try:
        from sklearn.model_selection import cross_validate
        from sklearn.ensemble import RandomForestClassifier
        
        # Create model
        model = RandomForestClassifier(random_state=42)
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=self.cpu_optimizer.get_optimal_worker_count() if self.cpu_optimizer else -1,
            return_train_score=True
        )
        
        results = {
            'cv_folds': cv_folds,
            'scoring': scoring,
            'test_scores': cv_results[f'test_{scoring}'],
            'train_scores': cv_results[f'train_{scoring}'],
            'mean_test_score': cv_results[f'test_{scoring}'].mean(),
            'std_test_score': cv_results[f'test_{scoring}'].std(),
            'mean_train_score': cv_results[f'train_{scoring}'].mean(),
            'std_train_score': cv_results[f'train_{scoring}'].std(),
            'fit_times': cv_results.get('fit_time', []),
            'score_times': cv_results.get('score_time', [])
        }
        
        _safe_print(f"✅ Cross-validation completed: {scoring} = {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
        return results
        
    except Exception as e:
        self.logger.error(f"Cross-validation failed: {e}")
        raise

def lookahead_validation(self, X: np.ndarray, y: np.ndarray,
                       lookahead_steps: int = 10,
                       train_size: float = 0.8) -> Dict[str, Any]:
    """
    Perform lookahead validation for time series data.
    
    Args:
        X: Features
        y: Target variable
        lookahead_steps: Number of steps to look ahead
        train_size: Training set size
        
    Returns:
        Lookahead validation results
    """
    _safe_print(f"👀 Performing lookahead validation ({lookahead_steps} steps)...")
    
    try:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        model = RandomForestClassifier(random_state=42)
        scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict with lookahead
            y_pred = model.predict(X_test)
            
            # Calculate score
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
            
            _safe_print(f"Fold {fold + 1}: {score:.4f}")
        
        results = {
            'lookahead_steps': lookahead_steps,
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_folds': len(scores)
        }
        
        _safe_print(f"✅ Lookahead validation completed: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        return results
        
    except Exception as e:
        self.logger.error(f"Lookahead validation failed: {e}")
        raise

# Factory function and demo
def create_optimized_trainer(config: Optional[TrainingConfig] = None) -> OptimizedTrainer:
    """
    Factory function to create OptimizedTrainer instance.
    
    Args:
        config: Training configuration
        
    Returns:
        OptimizedTrainer instance
    """
    return OptimizedTrainer(config)

def demo_optimized_trainer():
    """Demonstrate OptimizedTrainer functionality."""
    _safe_print("🚀 OptimizedTrainer Demo")
    _safe_print("=" * 50)
    
    try:
        # Create configuration
        config = TrainingConfig(
            max_epochs=10,
            batch_size=64,
            learning_rate=0.001,
            enable_gpu=True,
            enable_memory_optimization=True,
            enable_parallel=True,
            output_dir="demo_outputs"
        )
        
        # Create trainer
        trainer = OptimizedTrainer(config)
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.random.randint(0, 2, 1000)
        
        _safe_print(f"📊 Generated sample data: {X.shape}, {y.shape}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
        
        # Demo hyperparameter optimization
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        _safe_print("🔍 Testing hyperparameter optimization...")
        opt_results = trainer.hyperparameter_optimization(
            X_train, y_train, param_grid, method='grid', cv_folds=3
        )
        
        _safe_print(f"✅ Best parameters: {opt_results['best_params']}")
        _safe_print(f"✅ Best score: {opt_results['best_score']:.4f}")
        
        # Demo cross-validation
        _safe_print("🔄 Testing cross-validation...")
        cv_results = trainer.cross_validate(X_train, y_train, cv_folds=3)
        _safe_print(f"✅ CV Score: {cv_results['mean_test_score']:.4f} ± {cv_results['std_test_score']:.4f}")
        
        # Demo lookahead validation
        _safe_print("👀 Testing lookahead validation...")
        lookahead_results = trainer.lookahead_validation(X_train, y_train, lookahead_steps=5)
        _safe_print(f"✅ Lookahead Score: {lookahead_results['mean_score']:.4f} ± {lookahead_results['std_score']:.4f}")
        
        # Get performance report
        report = trainer.get_performance_report()
        _safe_print("📊 Performance Report:")
        _safe_print(f"  - Hardware: {report.get('hardware_info', {})}")
        _safe_print(f"  - Training Stats: {report.get('training_stats', {})}")
        
        # Cleanup
        trainer.cleanup()
        
        _safe_print("✅ Demo completed successfully!")
        
    except Exception as e:
        _safe_print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_optimized_trainer()