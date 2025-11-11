#!/usr/bin/env python3
"""
Script pour corriger automatiquement les imports incorrects de tprint dans tout le projet.

Remplace: from src.utils.tprint import tprint
Par:     from src.utils.tprint import tprint
"""

import os
import re
import sys
from pathlib import Path

def find_python_files_with_incorrect_imports(root_dir):
    """
    Trouve tous les fichiers Python contenant des imports incorrects de tprint.
    
    Args:
        root_dir (str): Répertoire racine pour la recherche
        
    Returns:
        list: Liste des chemins de fichiers avec des imports incorrects
    """
    files_with_issues = []
    root_path = Path(root_dir)
    
    # Parcourir récursivement tous les fichiers Python
    for py_file in root_path.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Vérifier si le fichier contient l'import incorrect
            if re.search(r'from\s+src\.printing\s+import\s+tprint', content):
                files_with_issues.append(str(py_file))
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {py_file}: {e}")
            
    return files_with_issues

def fix_imports_in_file(file_path):
    """
    Corrige les imports incorrects dans un fichier spécifique.
    
    Args:
        file_path (str): Chemin du fichier à corriger
        
    Returns:
        bool: True si des corrections ont été appliquées, False sinon
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remplacer l'import incorrect par le bon
        original_content = content
        content = re.sub(
            r'from\s+src\.printing\s+import\s+tprint',
            'from src.utils.tprint import tprint',
            content
        )
        
        # Vérifier si des modifications ont été faites
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
            
    except Exception as e:
        print(f"Erreur lors de la correction du fichier {file_path}: {e}")
        
    return False

def validate_import(file_path):
    """
    Vérifie qu'un fichier peut être importé sans erreur après correction.
    
    Args:
        file_path (str): Chemin du fichier à valider
        
    Returns:
        bool: True si l'import réussit, False sinon
    """
    try:
        # Convertir le chemin en module Python
        module_path = file_path.replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
            
        # Supprimer le préfixe 'src.' s'il existe
        if module_path.startswith('src.'):
            module_path = module_path[4:]
            
        # Essayer d'importer le module
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None:
            return False
            
        module = importlib.util.module_from_spec(spec)
        
        # Ne pas exécuter le module, juste vérifier la syntaxe
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
            
        return True
        
    except Exception as e:
        print(f"Erreur de validation pour {file_path}: {e}")
        return False

def main():
    """Fonction principale du script."""
    # Répertoire racine du projet
    root_dir = "src"
    
    print("🔍 Recherche des fichiers avec des imports incorrects de tprint...")
    files_with_issues = find_python_files_with_incorrect_imports(root_dir)
    
    if not files_with_issues:
        print("✅ Aucun fichier avec des imports incorrects trouvé.")
        return
        
    print(f"📋 {len(files_with_issues)} fichiers avec des imports incorrects trouvés:")
    for file_path in files_with_issues:
        print(f"  - {file_path}")
    
    print("\n🔧 Correction des imports...")
    corrected_files = []
    failed_files = []
    
    for file_path in files_with_issues:
        if fix_imports_in_file(file_path):
            corrected_files.append(file_path)
            print(f"✅ Corrigé: {file_path}")
        else:
            failed_files.append(file_path)
            print(f"❌ Échec: {file_path}")
    
    print(f"\n📊 Résumé des corrections:")
    print(f"  - Fichiers corrigés: {len(corrected_files)}")
    print(f"  - Fichiers en échec: {len(failed_files)}")
    
    if failed_files:
        print("\n⚠️ Fichiers qui n'ont pas pu être corrigés:")
        for file_path in failed_files:
            print(f"  - {file_path}")
    
    print("\n🔍 Validation des corrections...")
    validation_passed = []
    validation_failed = []
    
    for file_path in corrected_files:
        if validate_import(file_path):
            validation_passed.append(file_path)
        else:
            validation_failed.append(file_path)
    
    print(f"\n📊 Résumé de la validation:")
    print(f"  - Fichiers validés: {len(validation_passed)}")
    print(f"  - Fichiers en échec de validation: {len(validation_failed)}")
    
    if validation_failed:
        print("\n⚠️ Fichiers qui n'ont pas passé la validation:")
        for file_path in validation_failed:
            print(f"  - {file_path}")
    
    print(f"\n🎉 Opération terminée!")
    print(f"   Total des fichiers traités: {len(files_with_issues)}")
    print(f"   Corrections appliquées: {len(corrected_files)}")
    print(f"   Validations réussies: {len(validation_passed)}")

if __name__ == "__main__":
    main()