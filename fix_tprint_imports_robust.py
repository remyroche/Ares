#!/usr/bin/env python3
"""
Script robuste pour corriger les imports incorrects de tprint dans tout le projet.

Ce script gère les erreurs de syntaxe qui peuvent survenir lors de la modification des imports.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def find_files_with_incorrect_imports(root_dir: str = ".") -> List[str]:
    """Recherche tous les fichiers Python avec des imports incorrects de tprint."""
    files_with_issues = []
    
    for root, dirs, files in os.walk(root_dir):
        # Ignorer les répertoires de cache et les dépendances
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'from src.utils.tprint import tprint' in content:
                            files_with_issues.append(file_path)
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    return files_with_issues

def validate_syntax(content: str) -> Tuple[bool, Optional[str]]:
    """Valide la syntaxe Python du contenu."""
    try:
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def fix_imports_in_file(file_path: str) -> Dict[str, any]:
    """Corrige les imports dans un fichier spécifique avec validation."""
    result = {
        'file_path': file_path,
        'success': False,
        'error': None,
        'backup_created': False,
        'original_import_found': False,
        'correct_import_found': False
    }
    
    try:
        # Lire le contenu original
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Vérifier si l'import incorrect existe
        if 'from src.utils.tprint import tprint' not in original_content:
            result['success'] = True  # Pas de correction nécessaire
            result['error'] = "No incorrect import found"
            return result
        
        result['original_import_found'] = True
        
        # Vérifier si l'import correct existe déjà
        if 'from src.utils.tprint import tprint' in original_content:
            result['correct_import_found'] = True
        
        # Créer une sauvegarde
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        result['backup_created'] = True
        
        # Corriger l'import
        corrected_content = original_content.replace(
            'from src.utils.tprint import tprint',
            'from src.utils.tprint import tprint'
        )
        
        # Valider la syntaxe du contenu corrigé
        is_valid, error_msg = validate_syntax(corrected_content)
        
        if not is_valid:
            # Si la syntaxe est invalide, essayer une approche plus prudente
            print(f"⚠️ Erreur de syntaxe dans {file_path}: {error_msg}")
            print("🔄 Tentative de correction plus prudente...")
            
            # Approche plus prudente: supprimer d'abord l'import incorrect
            lines = original_content.split('\n')
            corrected_lines = []
            
            for line in lines:
                if 'from src.utils.tprint import tprint' in line:
                    # Remplacer uniquement cette ligne
                    corrected_line = line.replace('from src.utils.tprint import tprint', 'from src.utils.tprint import tprint')
                    corrected_lines.append(corrected_line)
                else:
                    corrected_lines.append(line)
            
            corrected_content = '\n'.join(corrected_lines)
            
            # Valider à nouveau
            is_valid, error_msg = validate_syntax(corrected_content)
            
            if not is_valid:
                result['error'] = f"Syntax error after correction: {error_msg}"
                return result
        
        # Écrire le contenu corrigé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(corrected_content)
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result

def restore_from_backup(file_path: str) -> bool:
    """Restaure un fichier à partir de sa sauvegarde."""
    backup_path = file_path + '.backup'
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_content = f.read()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(backup_content)
            
            os.remove(backup_path)
            return True
        except Exception:
            return False
    return False

def main():
    """Fonction principale du script."""
    print("🔍 Recherche des fichiers avec des imports incorrects de tprint...")
    
    # Trouver tous les fichiers avec des imports incorrects
    files_with_issues = find_files_with_incorrect_imports()
    
    if not files_with_issues:
        print("✅ Aucun fichier avec des imports incorrects trouvé.")
        return
    
    print(f"📁 {len(files_with_issues)} fichiers trouvés avec des imports incorrects")
    
    # Traiter chaque fichier
    success_count = 0
    error_count = 0
    syntax_error_files = []
    
    for file_path in files_with_issues:
        print(f"\n🔧 Traitement de: {file_path}")
        
        result = fix_imports_in_file(file_path)
        
        if result['success']:
            if result['original_import_found']:
                print(f"  ✅ Import corrigé avec succès")
                success_count += 1
            else:
                print(f"  ℹ️ Pas de correction nécessaire")
        else:
            print(f"  ❌ Erreur: {result['error']}")
            error_count += 1
            
            # Si c'est une erreur de syntaxe, noter le fichier pour traitement manuel
            if "Syntax error" in result.get('error', ''):
                syntax_error_files.append(file_path)
                # Restaurer depuis la sauvegarde
                if result['backup_created']:
                    restore_from_backup(file_path)
                    print(f"  🔄 Fichier restauré depuis la sauvegarde")
    
    # Résumé
    print(f"\n📊 Résumé du traitement:")
    print(f"  ✅ Fichiers corrigés: {success_count}")
    print(f"  ❌ Erreurs: {error_count}")
    
    if syntax_error_files:
        print(f"\n⚠️ Fichiers nécessitant une correction manuelle (erreurs de syntaxe):")
        for file_path in syntax_error_files:
            print(f"  - {file_path}")
        
        print(f"\n💡 Ces fichiers peuvent nécessiter une correction manuelle des erreurs de syntaxe")
        print(f"   qui ne sont pas liées aux imports de tprint.")

if __name__ == "__main__":
    main()