#!/usr/bin/env python3
"""
Script pour nettoyer les objets numpy scalars du fichier de configuration YAML.

Ce script résout l'erreur YAML/HPO en convertissant les objets numpy sérialisés
en types Python natifs qui peuvent être correctement sérialisés/désérialisés.
"""

import yaml
import numpy as np
import re
from pathlib import Path

def clean_numpy_scalars(obj):
    """
    Nettoie récursivement les objets numpy scalars d'une structure de données.
    
    Args:
        obj: Objet à nettoyer (dict, list, ou scalaire)
        
    Returns:
        Objet nettoyé avec les scalars numpy convertis en types natifs
    """
    if isinstance(obj, dict):
        return {k: clean_numpy_scalars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_scalars(item) for item in obj]
    elif isinstance(obj, np.generic):
        # Convertir les scalars numpy en types Python natifs
        if np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
        else:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Gérer les objets numpy avec attributs
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
        except (AttributeError, TypeError):
            return str(obj)
    else:
        return obj

def fix_yaml_file(file_path):
    """
    Corrige un fichier YAML en nettoyant les objets numpy scalars.
    
    Args:
        file_path: Chemin vers le fichier YAML à corriger
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Fichier introuvable: {file_path}")
        return False
    
    print(f"🔧 Nettoyage du fichier YAML: {file_path}")
    
    try:
        # Lire le fichier YAML avec gestion des erreurs
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Tenter de charger avec safe_load d'abord
        try:
            config = yaml.safe_load(content)
            print("✅ Fichier YAML chargé avec safe_load")
        except yaml.constructor.ConstructorError as e:
            print(f"⚠️  Erreur avec safe_load, tentative avec UnsafeLoader: {e}")
            # Utiliser UnsafeLoader pour les objets numpy existants
            config = yaml.load(content, Loader=yaml.UnsafeLoader)
            print("✅ Fichier YAML chargé avec UnsafeLoader")
        
        # Nettoyer les objets numpy
        cleaned_config = clean_numpy_scalars(config)
        print("✅ Objets numpy nettoyés")
        
        # Créer une sauvegarde
        backup_path = file_path.with_suffix('.yaml.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"💾 Sauvegarde créée: {backup_path}")
        
        # Écrire le fichier corrigé
        with open(file_path, 'w') as f:
            yaml.dump(cleaned_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Fichier YAML corrigé: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement du fichier {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale."""
    # Fichier YAML problématique identifié dans les logs
    yaml_file = "src/training/steps/model_training/analyst_base_config.yaml"
    
    print("🚀 Correction de l'erreur YAML/HPO")
    print("=" * 50)
    
    success = fix_yaml_file(yaml_file)
    
    if success:
        print("=" * 50)
        print("✅ Correction terminée avec succès!")
        print("📝 L'erreur 'could not determine a constructor for the tag' devrait être résolue")
        print("🔄 Le processus HPO devrait maintenant pouvoir mettre à jour le fichier YAML")
    else:
        print("=" * 50)
        print("❌ Échec de la correction")
        print("🔍 Vérifiez les erreurs ci-dessus")

if __name__ == "__main__":
    main()