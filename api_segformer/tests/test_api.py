# tests/test_api.py
import requests
import time
import json
from PIL import Image
import io
import os
import sys

API_URL = os.environ.get('API_URL', 'http://localhost:5000')

def test_health():
    """Test health check"""
    print("🧪 Test health check...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print(f"   ✅ API en bonne santé - Modèle: {data['model']}")
    return True

def test_prediction(image_path=None):
    """Test prediction avec une vraie image"""
    print("\n🧪 Test prédiction...")
    
    # Utiliser l'image fournie ou la première image de test
    if image_path and os.path.exists(image_path):
        print(f"   Image: {image_path}")
        image = Image.open(image_path)
    elif os.path.exists('test_images/test_000.png'):
        image_path = 'test_images/test_000.png'
        print(f"   Image par défaut: {image_path}")
        image = Image.open(image_path)
    else:
        print("   Création d'une image de test...")
        image = Image.new('RGB', (512, 256), color='blue')
    
    # Convertir en bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    files = {'image': ('test.png', buffer, 'image/png')}
    
    # Envoyer la requête
    start = time.time()
    response = requests.post(f"{API_URL}/predict", files=files)
    elapsed = (time.time() - start) * 1000
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    
    print(f"   ✅ Prédiction réussie")
    print(f"   - Temps total: {elapsed:.1f} ms")
    print(f"   - Temps inférence: {data['inference_time_ms']:.1f} ms")
    print(f"   - Modèle: {data['model']}")
    
    if 'class_distribution' in data:
        print("   - Classes détectées:")
        # Trier par pourcentage décroissant
        classes = sorted(data['class_distribution'].items(), 
                        key=lambda x: x[1], reverse=True)
        for cls, pct in classes[:5]:  # Top 5 classes
            if pct > 0.1:  # Afficher seulement si > 0.1%
                print(f"     • {cls}: {pct:.1f}%")
    
    # Sauvegarder le résultat si demandé
    if 'segmentation_image' in data and image_path:
        save_result(data['segmentation_image'], image_path)
    
    return True

def save_result(base64_image, original_path):
    """Sauvegarde le résultat de segmentation"""
    import base64
    
    # Créer le dossier test_results s'il n'existe pas
    os.makedirs('test_results', exist_ok=True)
    
    # Nom du fichier de sortie
    basename = os.path.basename(original_path).replace('.png', '')
    output_path = f"test_results/{basename}_segmented.png"
    
    # Décoder et sauvegarder
    img_data = base64.b64decode(base64_image)
    with open(output_path, 'wb') as f:
        f.write(img_data)
    
    print(f"   💾 Résultat sauvé: {output_path}")

def run_all_tests(image_path=None):
    """Exécute tous les tests"""
    print("="*50)
    print("TESTS API SEGFORMER P9")
    print("="*50)
    print(f"URL: {API_URL}\n")
    
    try:
        test_health()
        test_prediction(image_path)
        
        print("\n" + "="*50)
        print("✅ TOUS LES TESTS PASSÉS")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Prendre l'image depuis les arguments ou utiliser celle par défaut
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'test_images/test_000.png'
    success = run_all_tests(image_path)
    sys.exit(0 if success else 1)