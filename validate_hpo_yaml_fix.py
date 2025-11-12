#!/usr/bin/env python3
"""
Script de validation pour confirmer que l'erreur YAML/HPO est résolue.

Ce script simule le processus complet de mise à jour HPO pour valider que:
1. Le fichier YAML peut être chargé sans erreur
2. La mise à jour des paramètres fonctionne
3. Le fichier peut être sauvegardé correctement
4. Le processus HPO peut continuer normalement
"""

import yaml
import numpy as np
from pathlib import Path
from src.training.steps.model_training.hpo_config import YAMLConfigUpdater, HierarchicalOptimizationResult
from src.utils.logger import system_logger

def test_yaml_loading():
    """Test que le fichier YAML peut être chargé sans erreur."""
    print("🔍 Test 1: Chargement du fichier YAML")
    
    try:
        with open('src/training/steps/model_training/analyst_base_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Fichier YAML chargé avec succès via safe_load")
        return True, config
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False, None

def test_yaml_config_updater():
    """Test que YAMLConfigUpdater fonctionne correctement."""
    print("\n🔧 Test 2: Initialisation de YAMLConfigUpdater")
    
    try:
        updater = YAMLConfigUpdater('src/training/steps/model_training/analyst_base_config.yaml')
        print("✅ YAMLConfigUpdater initialisé avec succès")
        return True, updater
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return False, None

def test_numpy_cleaning(updater, config):
    """Test que le nettoyage des objets numpy fonctionne."""
    print("\n🧹 Test 3: Nettoyage des objets numpy")
    
    try:
        cleaned_config = updater._clean_numpy_scalars(config)
        print("✅ Nettoyage des objets numpy réussi")
        
        # Vérifier qu'il n'y a plus d'objets numpy problématiques
        has_numpy_objects = _check_for_numpy_objects(cleaned_config)
        if not has_numpy_objects:
            print("✅ Aucun objet numpy problématique détecté")
            return True, cleaned_config
        else:
            print("⚠️  Objets numpy encore présents après nettoyage")
            return False, cleaned_config
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        return False, None

def _check_for_numpy_objects(obj, path=""):
    """Vérifie récursivement s'il y a des objets numpy problématiques."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _check_for_numpy_objects(v, f"{path}.{k}" if path else k):
                return True
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if _check_for_numpy_objects(item, f"{path}[{i}]"):
                return True
    elif hasattr(obj, '__class__') and 'numpy' in str(type(obj)):
        print(f"⚠️  Objet numpy détecté à {path}: {type(obj)}")
        return True
    return False

def test_parameter_update(updater):
    """Test la mise à jour des paramètres HPO."""
    print("\n📝 Test 4: Mise à jour des paramètres HPO")
    
    try:
        # Créer des paramètres de test
        test_params = {
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 200
        }
        
        # Créer un résultat HPO factice
        test_result = HierarchicalOptimizationResult(
            best_params=test_params,
            best_score=0.85,
            group_results=[],
            total_time=120.5,
            total_trials=50
        )
        
        # Tenter la mise à jour
        success = updater.update_model_params(
            model_name='lgbm',
            optimal_params=test_params,
            hpo_result=test_result,
            model_path='analyst_config.base_models.lgbm'
        )
        
        if success:
            print("✅ Mise à jour des paramètres réussie")
            return True
        else:
            print("❌ Échec de la mise à jour des paramètres")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yaml_roundtrip():
    """Test que le fichier peut être sauvegardé et rechargé."""
    print("\n🔄 Test 5: Cycle de sauvegarde/rechargement")
    
    try:
        # Lire le fichier actuel
        with open('src/training/steps/model_training/analyst_base_config.yaml', 'r') as f:
            original_content = f.read()
        
        # Parser et resauvegarder
        config = yaml.safe_load(original_content)
        with open('src/training/steps/model_training/analyst_base_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Recharger et comparer
        with open('src/training/steps/model_training/analyst_base_config.yaml', 'r') as f:
            reloaded_content = f.read()
        
        # Vérifier que les contenus sont équivalents
        reloaded_config = yaml.safe_load(reloaded_content)
        
        # Comparer les structures (pas nécessairement le formatage exact)
        if _compare_configs(config, reloaded_config):
            print("✅ Cycle de sauvegarde/rechargement réussi")
            return True
        else:
            print("❌ Incohérence détectée dans le cycle")
            return False
    except Exception as e:
        print(f"❌ Erreur lors du cycle: {e}")
        return False

def _compare_configs(config1, config2):
    """Compare deux structures de configuration de manière récursive."""
    if type(config1) != type(config2):
        return False
    
    if isinstance(config1, dict):
        if set(config1.keys()) != set(config2.keys()):
            return False
        for key in config1:
            if not _compare_configs(config1[key], config2[key]):
                return False
    elif isinstance(config1, list):
        if len(config1) != len(config2):
            return False
        for i in range(len(config1)):
            if not _compare_configs(config1[i], config2[i]):
                return False
    else:
        return config1 == config2
    
    return True

def test_hpo_integration():
    """Test l'intégration complète avec le système HPO."""
    print("\n🚀 Test 6: Intégration HPO complète")
    
    try:
        # Importer et tester les composants HPO
        from src.training.steps.model_training.hpo_config import HPOOrchestrator
        
        orchestrator = HPOOrchestrator(
            config_file='src/training/steps/model_training/analyst_base_config.yaml',
            execution_mode='light'
        )
        
        print("✅ HPOOrchestrator initialisé avec succès")
        
        # Tester l'obtention des groupes de paramètres
        lgbm_groups = orchestrator.get_parameter_groups('lgbm')
        if lgbm_groups:
            print(f"✅ Groupes de paramètres LGBM obtenus: {len(lgbm_groups)} groupes")
            return True
        else:
            print("❌ Échec de l'obtention des groupes de paramètres")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de l'intégration HPO: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de validation."""
    print("🚀 Validation de la correction YAML/HPO")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Chargement YAML
    success, config = test_yaml_loading()
    if success:
        tests_passed += 1
    
    # Test 2: Initialisation YAMLConfigUpdater
    success, updater = test_yaml_config_updater()
    if success:
        tests_passed += 1
    
    # Test 3: Nettoyage numpy
    if updater and config:
        success, _ = test_numpy_cleaning(updater, config)
        if success:
            tests_passed += 1
    
    # Test 4: Mise à jour des paramètres
    if updater:
        success = test_parameter_update(updater)
        if success:
            tests_passed += 1
    
    # Test 5: Cycle de sauvegarde/rechargement
    success = test_yaml_roundtrip()
    if success:
        tests_passed += 1
    
    # Test 6: Intégration HPO
    success = test_hpo_integration()
    if success:
        tests_passed += 1
    
    # Résultats
    print("\n" + "=" * 60)
    print(f"📊 RÉSULTATS: {tests_passed}/{total_tests} tests réussis")
    
    if tests_passed == total_tests:
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ L'erreur YAML/HPO est complètement résolue")
        print("🔄 Le processus HPO peut maintenant fonctionner normalement")
        return True
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔍 Des problèmes peuvent subsister")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)