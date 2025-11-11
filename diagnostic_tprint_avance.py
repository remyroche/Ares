#!/usr/bin/env python3

"""
Diagnostic avancé pour comprendre l'import tprint
"""

import sys
import os
from pathlib import Path
import importlib.util

def main():
    print("🔍 Diagnostic avancé des imports tprint...")
    print("=" * 60)
    
    # 1. Vérifier le répertoire src/printing
    print("\n📁 1. Vérification du répertoire src/printing:")
    src_printing_path = Path("src/printing")
    if src_printing_path.exists():
        print(f"✅ Répertoire src/printing existe: {src_printing_path}")
        print("📋 Contenu:")
        for item in src_printing_path.iterdir():
            print(f"   - {item.name}")
    else:
        print(f"❌ Répertoire src/printing n'existe pas: {src_printing_path}")
    
    # 2. Vérifier le répertoire src/utils
    print("\n📁 2. Vérification du répertoire src/utils:")
    src_utils_path = Path("src/utils")
    if src_utils_path.exists():
        print(f"✅ Répertoire src/utils existe: {src_utils_path}")
        tprint_path = src_utils_path / "tprint.py"
        if tprint_path.exists():
            print(f"✅ Fichier tprint.py existe: {tprint_path}")
        else:
            print(f"❌ Fichier tprint.py n'existe pas: {tprint_path}")
    else:
        print(f"❌ Répertoire src/utils n'existe pas: {src_utils_path}")
    
    # 3. Tester l'import direct avec sys.modules
    print("\n🧪 3. Test d'import direct:")
    try:
        # Vérifier si src.printing est dans sys.modules
        if 'src.printing' in sys.modules:
            print("✅ src.printing est déjà dans sys.modules")
            module = sys.modules['src.printing']
            print(f"   Module: {module}")
            print(f"   Fichier: {getattr(module, '__file__', 'N/A')}")
        else:
            print("❌ src.printing n'est pas dans sys.modules")
        
        # Vérifier si src.utils.tprint est dans sys.modules
        if 'src.utils.tprint' in sys.modules:
            print("✅ src.utils.tprint est déjà dans sys.modules")
            module = sys.modules['src.utils.tprint']
            print(f"   Module: {module}")
            print(f"   Fichier: {getattr(module, '__file__', 'N/A')}")
        else:
            print("❌ src.utils.tprint n'est pas dans sys.modules")
    
    except Exception as e:
        print(f"❌ Erreur lors de la vérification sys.modules: {e}")
    
    # 4. Tester l'import avec importlib
    print("\n🧪 4. Test d'import avec importlib:")
    try:
        # Essayer d'importer src.printing
        spec = importlib.util.find_spec("src.printing")
        if spec:
            print(f"✅ Spec trouvé pour src.printing: {spec}")
            print(f"   Origin: {getattr(spec, 'origin', 'N/A')}")
            print(f"   Submodule search locations: {getattr(spec, 'submodule_search_locations', 'N/A')}")
        else:
            print("❌ Pas de spec trouvé pour src.printing")
        
        # Essayer d'importer src.utils.tprint
        spec = importlib.util.find_spec("src.utils.tprint")
        if spec:
            print(f"✅ Spec trouvé pour src.utils.tprint: {spec}")
            print(f"   Origin: {getattr(spec, 'origin', 'N/A')}")
        else:
            print("❌ Pas de spec trouvé pour src.utils.tprint")
    
    except Exception as e:
        print(f"❌ Erreur lors de la recherche de spec: {e}")
    
    # 5. Vérifier les métadonnées du package src
    print("\n📦 5. Vérification du package src:")
    try:
        spec = importlib.util.find_spec("src")
        if spec:
            print(f"✅ Spec trouvé pour src: {spec}")
            print(f"   Origin: {getattr(spec, 'origin', 'N/A')}")
            print(f"   Submodule search locations: {getattr(spec, 'submodule_search_locations', 'N/A')}")
            if hasattr(spec, 'loader'):
                print(f"   Loader: {spec.loader}")
        else:
            print("❌ Pas de spec trouvé pour src")
    except Exception as e:
        print(f"❌ Erreur lors de la recherche de spec src: {e}")
    
    # 6. Vérifier s'il y a des fichiers __init__.py qui créent des alias
    print("\n🔍 6. Recherche d'alias dans __init__.py:")
    src_init_path = Path("src/__init__.py")
    if src_init_path.exists():
        print(f"✅ Fichier src/__init__.py existe")
        with open(src_init_path, 'r') as f:
            content = f.read()
            if 'printing' in content:
                print("⚠️ Le mot 'printing' a été trouvé dans src/__init__.py")
                print("   Contenu pertinent:")
                for line_num, line in enumerate(content.split('\n'), 1):
                    if 'printing' in line.lower():
                        print(f"   Ligne {line_num}: {line.strip()}")
            else:
                print("✅ Pas de référence à 'printing' dans src/__init__.py")
    else:
        print("❌ Fichier src/__init__.py n'existe pas")
    
    # 7. Vérifier le sys.path
    print("\n🛤️ 7. Vérification du sys.path:")
    for i, path in enumerate(sys.path[:10]):  # Limiter aux 10 premiers
        print(f"   {i}: {path}")
        if i >= 9 and len(sys.path) > 10:
            print(f"   ... et {len(sys.path) - 10} autres chemins")
    
    # 8. Test d'import réel
    print("\n🧪 8. Test d'import réel:")
    try:
        print("   Tentative d'import: from src.printing import tprint")
        from src.printing import tprint
        print("✅ Import depuis src.printing réussi!")
        print(f"   tprint function: {tprint}")
        print(f"   Module: {tprint.__module__}")
        print(f"   File: {getattr(tprint, '__file__', 'N/A')}")
    except ImportError as e:
        print(f"❌ Import depuis src.printing échoué: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue lors de l'import depuis src.printing: {e}")
    
    try:
        print("   Tentative d'import: from src.utils.tprint import tprint")
        from src.utils.tprint import tprint
        print("✅ Import depuis src.utils.tprint réussi!")
        print(f"   tprint function: {tprint}")
        print(f"   Module: {tprint.__module__}")
        print(f"   File: {getattr(tprint, '__file__', 'N/A')}")
    except ImportError as e:
        print(f"❌ Import depuis src.utils.tprint échoué: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue lors de l'import depuis src.utils.tprint: {e}")

if __name__ == "__main__":
    main()