"""
Utilitaires pour gérer les problèmes de compatibilité SHAP avec CalibratedClassifierCV et NumPy
"""
import numpy as np
import warnings
from typing import Any, Optional, Union

def apply_numpy_shap_fix() -> bool:
    """
    Applique un patch temporaire pour corriger l'incompatibilité SHAP-NumPy
    
    SHAP utilise np.bool qui a été déprécié dans NumPy 1.20+
    Cette fonction crée un alias pour maintenir la compatibilité.
    
    Returns:
        bool: True si le patch a été appliqué avec succès
    """
    try:
        # Vérifier si np.bool existe déjà
        if not hasattr(np, 'bool'):
            # Créer un alias vers bool pour compatibilité avec SHAP
            np.bool = bool
            print("✅ [SHAP_UTILS] np.bool alias créé pour compatibilité SHAP")
            return True
        else:
            print("ℹ️ [SHAP_UTILS] np.bool existe déjà")
            return True
    except Exception as e:
        print(f"❌ [SHAP_UTILS] Erreur lors de l'application du patch: {e}")
        return False

def extract_base_model_from_calibrated(calibrated_model: Any) -> Optional[Any]:
    """
    Extrait le modèle de base d'un CalibratedClassifierCV
    
    Cette fonction gère différentes versions de scikit-learn et différents
    types de modèles calibrés pour extraire le modèle sous-jacent.
    
    Args:
        calibrated_model: Un modèle CalibratedClassifierCV
        
    Returns:
        Le modèle de base ou None si l'extraction échoue
    """
    try:
        # Vérifier si c'est bien un CalibratedClassifierCV
        if 'CalibratedClassifierCV' not in str(type(calibrated_model)):
            print(f"⚠️ [SHAP_UTILS] Le modèle n'est pas un CalibratedClassifierCV: {type(calibrated_model)}")
            return calibrated_model
        
        # Essayer différentes méthodes pour extraire le modèle de base
        base_model = None
        
        # Méthode 1: calibrated_model.estimators_[0] (scikit-learn récent)
        if hasattr(calibrated_model, 'estimators_') and len(calibrated_model.estimators_) > 0:
            base_model = calibrated_model.estimators_[0]
            print(f"✅ [SHAP_UTILS] Modèle de base extrait via estimators_[0]: {type(base_model)}")
            return base_model
        
        # Méthode 2: calibrated_model.estimator (versions plus anciennes)
        if hasattr(calibrated_model, 'estimator'):
            base_model = calibrated_model.estimator
            print(f"✅ [SHAP_UTILS] Modèle de base extrait via estimator: {type(base_model)}")
            return base_model
        
        # Méthode 3: calibrated_model.base_estimator (alternative)
        if hasattr(calibrated_model, 'base_estimator'):
            base_model = calibrated_model.base_estimator
            print(f"✅ [SHAP_UTILS] Modèle de base extrait via base_estimator: {type(base_model)}")
            return base_model
        
        # Méthode 4: Chercher dans les attributs privés
        for attr_name in ['_estimator', '_base_estimator', '_calibrated_classifiers']:
            if hasattr(calibrated_model, attr_name):
                attr_value = getattr(calibrated_model, attr_name)
                if attr_name == '_calibrated_classifiers' and hasattr(attr_value, '__iter__'):
                    # Pour les classificateurs calibrés, le premier élément contient le modèle
                    for calibrated_clf in attr_value:
                        if hasattr(calibrated_clf, 'estimator'):
                            base_model = calibrated_clf.estimator
                            print(f"✅ [SHAP_UTILS] Modèle de base extrait via _calibrated_classifiers: {type(base_model)}")
                            return base_model
                elif hasattr(attr_value, 'predict'):
                    base_model = attr_value
                    print(f"✅ [SHAP_UTILS] Modèle de base extrait via {attr_name}: {type(base_model)}")
                    return base_model
        
        print(f"❌ [SHAP_UTILS] Impossible d'extraire le modèle de base de: {type(calibrated_model)}")
        print(f"   Attributs disponibles: {[attr for attr in dir(calibrated_model) if not attr.startswith('__')]}")
        return None
        
    except Exception as e:
        print(f"❌ [SHAP_UTILS] Erreur lors de l'extraction du modèle de base: {e}")
        return None

def safe_shap_tree_explainer(model: Any, **kwargs) -> Any:
    """
    Crée un TreeExplainer SHAP de manière sécurisée
    
    Cette fonction gère les problèmes de compatibilité avec CalibratedClassifierCV
    et les problèmes de compatibilité NumPy.
    
    Args:
        model: Le modèle à expliquer
        **kwargs: Arguments supplémentaires pour TreeExplainer
        
    Returns:
        L'explainer SHAP ou None si la création échoue
    """
    try:
        # 1. Appliquer le patch NumPy si nécessaire
        apply_numpy_shap_fix()
        
        # 2. Extraire le modèle de base si c'est un CalibratedClassifierCV
        model_for_shap = extract_base_model_from_calibrated(model)
        if model_for_shap is None:
            model_for_shap = model
            print(f"⚠️ [SHAP_UTILS] Utilisation du modèle direct pour SHAP: {type(model_for_shap)}")
        
        # 3. Créer l'explainer
        import shap
        explainer = shap.TreeExplainer(model_for_shap, **kwargs)
        print(f"✅ [SHAP_UTILS] TreeExplainer créé avec succès pour: {type(model_for_shap)}")
        return explainer
        
    except Exception as e:
        print(f"❌ [SHAP_UTILS] Erreur lors de la création du TreeExplainer: {e}")
        return None

def safe_shap_values(explainer: Any, X: Any, **kwargs) -> Optional[Any]:
    """
    Calcule les valeurs SHAP de manière sécurisée
    
    Args:
        explainer: L'explainer SHAP
        X: Les données d'entrée
        **kwargs: Arguments supplémentaires pour shap_values
        
    Returns:
        Les valeurs SHAP ou None si le calcul échoue
    """
    try:
        # Appliquer le patch NumPy si nécessaire
        apply_numpy_shap_fix()
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X, **kwargs)
        print(f"✅ [SHAP_UTILS] Valeurs SHAP calculées avec succès")
        return shap_values
        
    except Exception as e:
        print(f"❌ [SHAP_UTILS] Erreur lors du calcul des valeurs SHAP: {e}")
        return None

def fallback_shap_explainer(model: Any, X: Any, **kwargs) -> Optional[Any]:
    """
    Fallback vers un explainer alternatif si TreeExplainer échoue
    
    Args:
        model: Le modèle à expliquer
        X: Les données d'entrée (pour KernelExplainer)
        **kwargs: Arguments supplémentaires
        
    Returns:
        L'explainer alternatif ou None si tout échoue
    """
    try:
        import shap
        
        # Essayer KernelExplainer (plus lent mais universel)
        if hasattr(model, 'predict_proba'):
            background_data = X[:min(100, len(X))]  # Limiter pour la performance
            explainer = shap.KernelExplainer(model.predict_proba, background_data, **kwargs)
            print(f"✅ [SHAP_UTILS] KernelExplainer créé comme fallback")
            return explainer
        
        print(f"❌ [SHAP_UTILS] Aucun explainer alternatif disponible pour: {type(model)}")
        return None
        
    except Exception as e:
        print(f"❌ [SHAP_UTILS] Erreur lors de la création du fallback explainer: {e}")
        return None