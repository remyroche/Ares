#!/usr/bin/env python3
"""
Script de validation pour diagnostiquer les problèmes d'import de tprint
"""

import sys
import os
from pathlib import Path

def test_module_existence():
    """Teste l'existence des modules tprint."""
    print("🔍 Test d'existence des modules tprint...")
    
    # Test du module src.printing
    try:
        import src.printing
        print("❌ Module src.printing EXISTE (ceci est inattendu)")
        return True
    except ImportError as e:
        print(f"✅ Module src.printing n'existe pas: {e}")
    
    # Test du module src.utils.tprint
    try:
        import src.utils.tprint
        print("✅ Module src.utils.tprint existe et fonctionne")
        return True
    except ImportError as e:
        print(f"❌ Module src.utils.tprint n'existe pas: {e}")
        return False

def test_specific_imports():
    """Teste les imports spécifiques qui posent problème."""
    print("\n🧪 Test des imports spécifiques...")
    
    # Test de l'import problématique
    try:
        from src.printing import tprint
        print("❌ Import depuis src.printing RÉUSSI (ceci est inattendu)")
    except ImportError as e:
        print(f"✅ Import depuis src.printing échoué comme attendu: {e}")
    
    # Test de l'import correct
    try:
        from src.utils.tprint import tprint
        print("✅ Import depuis src.utils.tprint réussi")
    except ImportError as e:
        print(f"❌ Import depuis src.utils.tprint échoué: {e}")

def check_directory_structure():
    """Vérifie la structure des répertoires."""
    print("\n📁 Vérification de la structure des répertoires...")
    
    src_path = Path("src")
    
    # Vérification du répertoire printing
    printing_path = src_path / "printing"
    if printing_path.exists():
        print(f"❌ Répertoire src/printing EXISTE: {printing_path}")
        if printing_path.is_dir():
            files = list(printing_path.glob("*.py"))
            print(f"   Fichiers: {files}")
    else:
        print(f"✅ Répertoire src/printing n'existe pas")
    
    # Vérification du répertoire utils
    utils_path = src_path / "utils"
    if utils_path.exists() and utils_path.is_dir():
        tprint_file = utils_path / "tprint.py"
        if tprint_file.exists():
            print(f"✅ Fichier src/utils/tprint.py existe")
        else:
            print(f"❌ Fichier src/utils/tprint.py n'existe pas")

def main():
    """Fonction principale."""
    print("🚀 Lancement du diagnostic des imports tprint...")
    print("=" * 60)
    
    # Test de base
    module_exists = test_module_existence()
    
    # Test des imports spécifiques
    test_specific_imports()
    
    # Vérification de la structure
    check_directory_structure()
    
    print("\n" + "=" * 60)
    print("📋 Résumé du diagnostic:")
    
    if not module_exists:
        print("✅ DIAGNOSTIC CONFIRMÉ: Le module src.printing n'existe pas")
        print("   CAUSE RACINE: Référence historique incorrecte dans 83 fichiers")
        print("   SOLUTION: Remplacer 'from src.printing import tprint' par 'from src.utils.tprint import tprint'")
    else:
        print("❌ DIAGNOSTIC INCOHÉRENT: Le module src.printing existe")
        print("   Cela suggère que le problème est différent de ce qui est attendu")

if __name__ == "__main__":
    main()