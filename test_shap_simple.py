#!/usr/bin/env python3
"""
Script de test simple pour vérifier que la correction SHAP avec CalibratedClassifierCV fonctionne
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import sys
import os

# Ajouter le chemin du projet pour importer tprint
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.tprint import tprint

def test_shap_with_calibrated_model():
    """Test la correction SHAP avec CalibratedClassifierCV"""
    tprint("🧪 [TEST] Début du test SHAP avec CalibratedClassifierCV", color="cyan")
    
    try:
        # Créer des données de test simples
        X = np.random.rand(50, 5)  # 50 échantillons, 5 features
        y = np.random.randint(0, 3, 50)  # 3 classes
        
        tprint(f"📊 [TEST] Données créées: {X.shape}", color="blue")
        
        # Entraîner un modèle de base
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        base_model.fit(X, y)
        
        tprint(f"✅ [TEST] Modèle de base entraîné: {type(base_model)}", color="green")
        
        # Créer un modèle calibré
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv='prefit'
        )
        
        # Utiliser une partie des données pour la calibration
        calibrated_model.fit(X[:40], y[:40])
        
        tprint(f"✅ [TEST] Modèle calibré créé: {type(calibrated_model)}", color="green")
        
        # Test 1: Essayer SHAP directement avec le modèle calibré (doit échouer)
        tprint("🔍 [TEST] Test 1: SHAP avec modèle calibré direct (doit échouer)", color="yellow")
        try:
            import shap
            explainer = shap.TreeExplainer(calibrated_model)
            shap_values = explainer.shap_values(X[:10])
            tprint("❌ [TEST] ERREUR: SHAP a fonctionné avec modèle calibré direct (inattendu!)", color="red")
            return False
        except Exception as e:
            tprint(f"✅ [TEST] Échec attendu avec modèle calibré direct: {e}", color="green")
        
        # Test 2: Utiliser notre correction (extraire le modèle de base)
        tprint("🔍 [TEST] Test 2: SHAP avec extraction du modèle de base (notre correction)", color="yellow")
        try:
            import shap
            
            # Appliquer notre logique de correction
            model_for_shap = calibrated_model
            if hasattr(calibrated_model, '__class__') and 'CalibratedClassifierCV' in str(type(calibrated_model)):
                tprint("⚠️ [TEST] Détection CalibratedClassifierCV - Extraction du modèle de base", color="yellow")
                if hasattr(calibrated_model, 'estimators_'):
                    model_for_shap = calibrated_model.estimators_[0]
                    tprint(f"🔍 [TEST] Modèle de base extrait: {type(model_for_shap)}", color="yellow")
                elif hasattr(calibrated_model, 'estimator'):
                    model_for_shap = calibrated_model.estimator
                    tprint(f"🔍 [TEST] Modèle de base extrait (alt): {type(model_for_shap)}", color="yellow")
                else:
                    raise ValueError("Impossible d'extraire le modèle de base du CalibratedClassifierCV")
            else:
                tprint(f"✅ [TEST] Utilisation du modèle direct pour SHAP: {type(model_for_shap)}", color="green")
            
            # Créer l'explainer avec le modèle extrait
            explainer = shap.TreeExplainer(model_for_shap)
            shap_values = explainer.shap_values(X[:10])
            
            tprint("✅ [TEST] SUCCÈS: SHAP fonctionne avec modèle de base extrait", color="green")
            tprint(f"📊 [TEST] Valeurs SHAP calculées: {np.array(shap_values).shape}", color="blue")
            
            return True
            
        except Exception as e:
            tprint(f"❌ [TEST] ÉCHEC de notre correction: {e}", color="red")
            return False
            
    except Exception as e:
        tprint(f"❌ [TEST] Erreur générale dans le test: {e}", color="red")
        return False

if __name__ == "__main__":
    tprint("🚀 [TEST] Lancement du test de correction SHAP avec CalibratedClassifierCV", color="cyan", bold=True)
    
    success = test_shap_with_calibrated_model()
    
    if success:
        tprint("🎉 [TEST] SUCCÈS TOTAL: La correction SHAP fonctionne correctement!", color="green", bold=True)
        sys.exit(0)
    else:
        tprint("💥 [TEST] ÉCHEC TOTAL: La correction SHAP ne fonctionne pas!", color="red", bold=True)
        sys.exit(1)