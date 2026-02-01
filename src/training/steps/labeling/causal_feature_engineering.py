"""
Causal Feature Engineering Module

Implements causal feature engineering techniques including:
1. Causal denoising using causal relationships
2. Causal adjustment of features
3. Causal imputation of missing values
4. Causal feature transformation

Key Features:
1. Causal denoising using discovered causal relationships
2. Causal feature adjustment based on parent effects
3. Causal imputation using structural equations
4. Causal feature transformation for enhanced prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class CausalFeatureEngineering:
    """
    Causal feature engineering using discovered causal relationships.
    
    Applies causal adjustments, denoising, and transformations to features
    based on their causal relationships for improved model performance.
    """
    
    def __init__(
        self,
        causal_graph: Optional[Dict[str, List[str]]] = None,
        causal_strength: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
        denoising_method: str = "ridge",
        imputation_method: str = "causal",
        verbose: bool = True
    ):
        """
        Initialize Causal Feature Engineering.
        
        Args:
            causal_graph: Causal graph from discovery
            causal_strength: Causal strength matrix
            variable_names: Variable names
            denoising_method: Method for causal denoising
            imputation_method: Method for causal imputation
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.causal_graph = causal_graph
        self.causal_strength = causal_strength
        self.variable_names = variable_names or []
        self.denoising_method = denoising_method
        self.imputation_method = imputation_method
        
        # Storage for learned models
        self.causal_models_ = {}
        self.causal_adjustments_ = {}
        self.feature_transformations_ = {}

    def _prune_to_data(self, data: pd.DataFrame) -> None:
        if data is None or data.empty:
            return
        available_cols = set(data.columns)

        if self.causal_graph:
            pruned_graph: Dict[str, List[str]] = {}
            for target_var, parent_vars in self.causal_graph.items():
                if target_var not in available_cols:
                    continue
                valid_parents = [p for p in parent_vars if p in available_cols]
                pruned_graph[target_var] = valid_parents
            self.causal_graph = pruned_graph

        if self.causal_models_:
            pruned_models: Dict[str, Dict[str, Any]] = {}
            for target_var, model_info in self.causal_models_.items():
                if target_var not in available_cols:
                    continue
                parents = [p for p in model_info.get('parents', []) if p in available_cols]
                if not parents:
                    continue
                updated_info = dict(model_info)
                updated_info['parents'] = parents
                pruned_models[target_var] = updated_info
            self.causal_models_ = pruned_models
        
    def learn_causal_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Learn causal relationships from data using linear models.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary of learned causal models
        """
        try:
            if self.verbose:
                tprint_info("🧠 Learning Causal Relationships: Starting...")
            
            if self.causal_graph is None:
                # User request: Fast fail, no correlation fallback
                tprint_error("   ❌ No causal graph provided for feature engineering")
                raise ValueError("Causal graph is required for causal feature engineering")
                # Create simple correlation-based graph
                numeric_data = data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                self.causal_graph = {}
                for var in numeric_data.columns:
                    # Find highly correlated variables as potential parents
                    correlations = corr_matrix[var].abs().sort_values(ascending=False)
                    parents = [corr for corr in correlations.index[1:6] if correlations[corr] > 0.3]
                    self.causal_graph[var] = parents
                
                tprint_info(f"   🔗 Created correlation-based graph with {len(self.causal_graph)} variables")
            
            # Learn causal models for each variable
            causal_models = {}
            models_trained = 0
            models_failed = 0
            
            for target_var, parent_vars in self.causal_graph.items():
                if target_var not in data.columns:
                    # Downgrade to info if it looks like a target/label variable (expected to be missing in X)
                    if any(x in target_var.lower() for x in ['target', 'label', 'ret_', 'bin_', 'meta_']):
                         if self.verbose:
                             pass # tprint_info(f"      ℹ️ Skipping causal engineering for label/target: {target_var}")
                    else:
                        if self.verbose:
                             tprint_info(f"   ℹ️ Variable {target_var} in graph but not in data (skipping)")
                    continue
                
                # Get parent variables that exist in data
                valid_parents = [p for p in parent_vars if p in data.columns]
                
                if len(valid_parents) > 0:
                    if self.verbose:
                        tprint_info(f"   📊 Learning model for {target_var} (parents: {valid_parents})")
                    
                    # Learn causal model
                    if self.denoising_method == "ridge":
                        model = Ridge(alpha=1.0)
                    elif self.denoising_method == "random_forest":
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                    else:
                        model = LinearRegression()
                    
                    try:
                        X = data[valid_parents].fillna(data[valid_parents].mean())
                        y = data[target_var].fillna(data[target_var].mean())
                        
                        model.fit(X, y)
                        causal_models[target_var] = {
                            'model': model,
                            'parents': valid_parents,
                            'coefficients': model.coef_ if hasattr(model, 'coef_') else None,
                            'score': model.score(X, y)
                        }
                        
                        models_trained += 1
                        if self.verbose and models_trained <= 5:  # Show first few
                            tprint_info(f"      ✅ {target_var}: score={model.score(X, y):.4f}")
                            
                    except Exception as e:
                        models_failed += 1
                        if self.verbose:
                            tprint_warning(f"      ❌ {target_var}: {e}")
                else:
                    if self.verbose:
                        tprint_warning(f"   ⚠️ No valid parents for {target_var}")
            
            self.causal_models_ = causal_models
            
            if self.verbose:
                tprint_success(f"✅ Causal relationship learning complete:")
                tprint_info(f"   - Models trained: {models_trained}")
                tprint_info(f"   - Models failed: {models_failed}")
                tprint_info(f"   - Success rate: {models_trained/(models_trained+models_failed)*100:.1f}%")
            
            return causal_models
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal relationship learning failed: {e}")
            raise
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal relationship learning failed: {e}")
            raise
    
    def causal_denoise(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal denoising to features.
        
        Args:
            data: Input data
            
        Returns:
            Denoised data
        """
        try:
            if self.verbose:
                tprint_info("🔧 Causal Denoising: Starting denoising process...")

            self._prune_to_data(data)
            
            if not self.causal_models_:
                if self.verbose:
                    tprint_info("   ⚙️ No causal models available, learning relationships...")
                self.learn_causal_relationships(data)

            self._prune_to_data(data)
            
            if not self.causal_models_:
                if self.verbose:
                    tprint_warning("   ⚠️ No causal models available, returning original data")
                return data
            
            denoised_data = data.copy()
            features_denoised = 0
            features_failed = 0

            missing_targets = [
                target_var for target_var in self.causal_models_.keys()
                if target_var not in data.columns
            ]
            if self.verbose and missing_targets:
                preview = ", ".join(missing_targets[:6])
                tprint_info(
                    "   ℹ️ Skipping causal denoising for missing targets: "
                    f"{preview}{'...' if len(missing_targets) > 6 else ''}"
                )

            for target_var, model_info in self.causal_models_.items():
                if target_var not in data.columns:
                    continue
                
                try:
                    if self.verbose and features_denoised < 3:  # Show first few
                        tprint_info(f"   📊 Denoising {target_var}...")
                    
                    model = model_info['model']
                    parents = model_info['parents']
                    
                    # Predict based on parents
                    X_parents = data[parents].fillna(data[parents].mean())
                    predicted = model.predict(X_parents)
                    
                    # Create denoised version (weighted average of actual and predicted)
                    actual = data[target_var].fillna(data[target_var].mean())
                    model_score = model_info['score']
                    
                    # Weight by model performance
                    weight = max(0.0, min(1.0, model_score))
                    denoised = weight * predicted + (1 - weight) * actual
                    
                    denoised_data[f"{target_var}_causal_denoised"] = denoised
                    features_denoised += 1
                    
                    if self.verbose and features_denoised <= 3:  # Show first few
                        tprint_info(f"      ✅ {target_var}: weight={weight:.3f}, score={model_score:.4f}")
                    
                except Exception as e:
                    features_failed += 1
                    if self.verbose:
                        tprint_warning(f"      ❌ {target_var}: {e}")
            
            if self.verbose:
                tprint_success(f"✅ Causal denoising complete:")
                tprint_info(f"   - Features denoised: {features_denoised}")
                tprint_info(f"   - Features failed: {features_failed}")
                tprint_info(f"   - Success rate: {features_denoised/(features_denoised+features_failed)*100:.1f}%")
            
            return denoised_data
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal denoising failed: {e}")
            return data
    
    def causal_adjustment(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal adjustment to remove parent effects.
        
        Args:
            data: Input data
            
        Returns:
            Causally adjusted data
        """
        try:
            if self.verbose:
                tprint_info("⚖️ Applying Causal Adjustment...")

            self._prune_to_data(data)
            
            if not self.causal_models_:
                self.learn_causal_relationships(data)

            self._prune_to_data(data)
            
            adjusted_data = data.copy()
            
            for target_var, model_info in self.causal_models_.items():
                if target_var not in data.columns:
                    continue
                
                model = model_info['model']
                parents = model_info['parents']
                
                # Calculate parent effects
                X_parents = data[parents].fillna(data[parents].mean())
                parent_effects = model.predict(X_parents)
                
                # Remove parent effects (residualize)
                actual = data[target_var].fillna(data[target_var].mean())
                adjusted = actual - parent_effects
                
                adjusted_data[f"{target_var}_causal_adjusted"] = adjusted
                
                # Store adjustment info
                self.causal_adjustments_[target_var] = {
                    'parent_effects_mean': np.mean(parent_effects),
                    'adjustment_mean': np.mean(adjusted),
                    'adjustment_std': np.std(adjusted)
                }
                
                if self.verbose and len(self.causal_models_) <= 5:
                    tprint_info(f"   - Adjusted {target_var} (parent effects removed)")
            
            if self.verbose:
                n_adjusted = len([col for col in adjusted_data.columns if "_causal_adjusted" in col])
                tprint_success(f"✅ Causal adjustment complete: {n_adjusted} features")
            
            return adjusted_data
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal adjustment failed: {e}")
            return data
    
    def causal_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal imputation for missing values.
        
        Args:
            data: Input data with missing values
            
        Returns:
            Imputed data
        """
        try:
            if self.verbose:
                tprint_info("🔍 Applying Causal Imputation...")

            self._prune_to_data(data)
            
            if not self.causal_models_:
                self.learn_causal_relationships(data)
            
            imputed_data = data.copy()
            
            # Count missing values
            missing_counts = data.isnull().sum()
            vars_with_missing = missing_counts[missing_counts > 0].index.tolist()
            
            if self.verbose:
                tprint_info(f"   - Variables with missing values: {len(vars_with_missing)}")
            
            for target_var in vars_with_missing:
                if target_var not in self.causal_models_:
                    continue
                
                model_info = self.causal_models_[target_var]
                model = model_info['model']
                parents = model_info['parents']
                
                # Find missing indices
                missing_mask = data[target_var].isnull()
                
                if missing_mask.sum() == 0:
                    continue
                
                # Use parents to impute
                X_parents = data[parents].fillna(data[parents].mean())
                predicted_values = model.predict(X_parents[missing_mask])
                
                # Impute missing values
                imputed_data.loc[missing_mask, target_var] = predicted_values
                
                if self.verbose and len(vars_with_missing) <= 5:
                    tprint_info(f"   - Imputed {missing_mask.sum()} values for {target_var}")
            
            if self.verbose:
                remaining_missing = imputed_data.isnull().sum().sum()
                tprint_success(f"✅ Causal imputation complete: {remaining_missing} missing values remaining")
            
            return imputed_data
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal imputation failed: {e}")
            return data
    
    def causal_feature_transformation(self, data: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
        """
        Apply causal feature transformations.
        
        Args:
            data: Input data
            max_features: Maximum new features to create (optimization)
            
        Returns:
            Transformed data
        """
        try:
            if self.verbose:
                tprint_info("🔄 Applying Causal Feature Transformations...")

            self._prune_to_data(data)
            
            if not self.causal_models_:
                self.learn_causal_relationships(data)
            
            # Optimization: Skip if too many models - transformation is expensive
            if len(self.causal_models_) > 30:
                tprint_info(f"   📉 Optimization: Limiting to top 30 models for transformation (had {len(self.causal_models_)})")
                # Take models with best scores
                sorted_models = sorted(self.causal_models_.items(), 
                                       key=lambda x: x[1].get('score', 0), reverse=True)[:30]
                models_to_process = dict(sorted_models)
            else:
                models_to_process = self.causal_models_
            
            transformed_data = data.copy()
            features_added = 0
            
            # Create causal interaction features
            for target_var, model_info in models_to_process.items():
                if features_added >= max_features:
                    break
                    
                if target_var not in data.columns:
                    continue
                
                parents = model_info['parents']
                
                # Create parent-target interactions (limit to top 2 parents only)
                for parent in parents[:2]:  # Reduced from 3 to 2
                    if features_added >= max_features:
                        break
                    if parent in data.columns:
                        interaction_name = f"{target_var}_x_{parent}_causal"
                        # Direct vectorized multiplication
                        transformed_data[interaction_name] = (
                            data[target_var].fillna(0).values * 
                            data[parent].fillna(0).values
                        )
                        features_added += 1
                
                # Create causal residual features (only for top models)
                if len(parents) > 0 and features_added < max_features and model_info.get('score', 0) > 0.1:
                    model = model_info['model']
                    try:
                        X_parents = data[parents].fillna(0)
                        predicted = model.predict(X_parents)
                        residual = data[target_var].fillna(0) - predicted
                        transformed_data[f"{target_var}_causal_residual"] = residual
                        features_added += 1
                    except Exception:
                        pass  # Skip on error, don't log each failure
            
            if self.verbose:
                tprint_success(f"✅ Causal transformation complete: {features_added} new features")
            
            return transformed_data
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal feature transformation failed: {e}")
            return data

    
    def apply_causal_engineering(
        self,
        data: pd.DataFrame,
        apply_denoising: bool = True,
        apply_adjustment: bool = True,
        apply_imputation: bool = True,
        apply_transformation: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply complete causal engineering pipeline.
        
        Args:
            data: Input data
            apply_denoising: Whether to apply causal denoising
            apply_adjustment: Whether to apply causal adjustment
            apply_imputation: Whether to apply causal imputation
            apply_transformation: Whether to apply feature transformation
            
        Returns:
            Tuple of (engineered_data, metadata)
        """
        try:
            if self.verbose:
                tprint_info("🚀 Starting Causal Feature Engineering Pipeline...")
            
            engineered_data = data.copy()
            self._prune_to_data(engineered_data)
            metadata = {
                'original_features': list(data.columns),
                'applied_steps': [],
                'feature_counts': {
                    'original': len(data.columns),
                    'final': len(data.columns),
                    'added': 0
                },
                'causal_models_count': 0
            }
            
            # Step 1: Learn causal relationships
            if not self.causal_models_:
                causal_models = self.learn_causal_relationships(data)
                metadata['causal_models_count'] = len(causal_models)
                metadata['applied_steps'].append('causal_relationship_learning')
            
            # Step 2: Causal imputation (if needed)
            if apply_imputation and data.isnull().sum().sum() > 0:
                engineered_data = self.causal_imputation(engineered_data)
                metadata['applied_steps'].append('causal_imputation')
            
            # Step 3: Causal denoising
            if apply_denoising:
                engineered_data = self.causal_denoise(engineered_data)
                metadata['applied_steps'].append('causal_denoising')
            
            # Step 4: Causal adjustment
            if apply_adjustment:
                engineered_data = self.causal_adjustment(engineered_data)
                metadata['applied_steps'].append('causal_adjustment')
            
            # Step 5: Feature transformation
            if apply_transformation:
                engineered_data = self.causal_feature_transformation(engineered_data)
                metadata['applied_steps'].append('causal_transformation')
            
            # Compile metadata
            metadata['final_features'] = list(engineered_data.columns)
            metadata['feature_counts'] = {
                'original': len(data.columns),
                'final': len(engineered_data.columns),
                'added': len(engineered_data.columns) - len(data.columns)
            }
            metadata['causal_adjustments'] = self.causal_adjustments_
            
            if self.verbose:
                tprint_success("✅ Causal Feature Engineering Complete:")
                tprint_info(f"   - Original features: {metadata['feature_counts']['original']}")
                tprint_info(f"   - Final features: {metadata['feature_counts']['final']}")
                tprint_info(f"   - Added features: {metadata['feature_counts']['added']}")
                tprint_info(f"   - Applied steps: {', '.join(metadata['applied_steps'])}")
            
            return engineered_data, metadata
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Causal feature engineering failed: {e}")
            return data, {'error': str(e)}
    
    def get_causal_summary(self) -> Dict[str, Any]:
        """
        Get summary of causal engineering results.
        
        Returns:
            Summary dictionary
        """
        return {
            'causal_models_count': len(self.causal_models_),
            'causal_adjustments_count': len(self.causal_adjustments_),
            'denoising_method': self.denoising_method,
            'imputation_method': self.imputation_method,
            'has_causal_graph': self.causal_graph is not None,
            'variable_names': self.variable_names
        }

# Convenience functions
def quick_causal_engineering(
    data: pd.DataFrame,
    causal_graph: Optional[Dict[str, List[str]]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick causal engineering with default parameters.
    
    Args:
        data: Input data
        causal_graph: Causal graph
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (engineered_data, metadata)
    """
    engineer = CausalFeatureEngineering(causal_graph=causal_graph, **kwargs)
    return engineer.apply_causal_engineering(data)

def causal_denoise_only(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Apply only causal denoising.
    
    Args:
        data: Input data
        **kwargs: Additional parameters
        
    Returns:
        Denoised data
    """
    engineer = CausalFeatureEngineering(**kwargs)
    return engineer.causal_denoise(data)

def causal_adjustment_only(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Apply only causal adjustment.
    
    Args:
        data: Input data
        **kwargs: Additional parameters
        
    Returns:
        Adjusted data
    """
    engineer = CausalFeatureEngineering(**kwargs)
    return engineer.causal_adjustment(data)
