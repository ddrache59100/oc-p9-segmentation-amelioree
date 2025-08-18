#!/bin/bash
# prepare_deployment_all.sh - Prépare le package avec TOUS les modèles pour Azure

echo "📦 Préparation du déploiement Azure avec TOUS les modèles SegFormer..."

# Nettoyer et créer le dossier
rm -rf azure_deploy_all
mkdir -p azure_deploy_all/{src,models,test_images,tests}

# Créer la structure des modèles
mkdir -p azure_deploy_all/models/{segformer_b0,segformer_b1,segformer_b2}

# Copier l'application
echo "📄 Copie de l'application..."
cp src/app.py azure_deploy_all/src/
cp src/config_azure.py azure_deploy_all/src/config.py
cp src/config_azure.py azure_deploy_all/src/config_azure.py
touch azure_deploy_all/src/__init__.py
cp tests/test_api.py azure_deploy_all/tests/
cp tests/compare_results.py azure_deploy_all/tests/
cp tests/benchmark_complete.py azure_deploy_all/tests/

# Copier TOUS les modèles
echo "🤖 Copie de tous les modèles..."

# B0
if [ -f "models/segformer_b0/model.onnx" ]; then
    cp models/segformer_b0/model.onnx azure_deploy_all/models/segformer_b0/
    echo "   ✓ B0 FP32 (14MB)"
fi
if [ -f "models/segformer_b0/model_quantized.onnx" ]; then
    cp models/segformer_b0/model_quantized.onnx azure_deploy_all/models/segformer_b0/
    echo "   ✓ B0 INT8 (4.3MB)"
fi

# B1
if [ -f "models/segformer_b1/model.onnx" ]; then
    cp models/segformer_b1/model.onnx azure_deploy_all/models/segformer_b1/
    echo "   ✓ B1 FP32 (53MB)"
fi
if [ -f "models/segformer_b1/model_quantized.onnx" ]; then
    cp models/segformer_b1/model_quantized.onnx azure_deploy_all/models/segformer_b1/
    echo "   ✓ B1 INT8 (14MB)"
fi

# B2
if [ -f "models/segformer_b2/model.onnx" ]; then
    cp models/segformer_b2/model.onnx azure_deploy_all/models/segformer_b2/
    echo "   ✓ B2 FP32 (105MB)"
fi
if [ -f "models/segformer_b2/model_quantized.onnx" ]; then
    cp models/segformer_b2/model_quantized.onnx azure_deploy_all/models/segformer_b2/
    echo "   ✓ B2 INT8 (28MB)"
fi

# Copier les images de test
echo "🖼️  Copie des images de test..."
if [ -d "test_images" ]; then
    cp -r test_images/* azure_deploy_all/test_images/
    echo "   ✓ $(ls test_images/*.png | wc -l) images copiées"
else
    echo "   ⚠️  Dossier test_images non trouvé"
fi

# Requirements avec transformers
echo "📝 Création des requirements..."
cat > azure_deploy_all/requirements.txt << EOFREQ
# requirements.txt - API SegFormer P9 (Azure Production)
Flask==2.3.2
flask-cors==4.0.0
Pillow==10.0.0
numpy==1.24.3
onnxruntime==1.22.0      # Version récente pour INT8
transformers==4.30.2      # Nécessaire pour FeatureExtractor
gunicorn==21.2.0
requests==2.31.0          # Optionnel, mais petit
EOFREQ

# Startup pour Azure
echo "🚀 Configuration startup..."
echo "gunicorn --bind=0.0.0.0 --timeout 600 --workers=1 src.app:app" > azure_deploy_all/startup.txt

# Point d'entrée pour Azure (app.py à la racine)
cat > azure_deploy_all/app.py << EOFAPP
# Point d'entrée pour Azure App Service
import sys
import os

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Importer l'app Flask depuis src/app.py
from src.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
EOFAPP

# Afficher la taille
SIZE=$(du -sh azure_deploy_all | cut -f1)
echo ""
echo "✅ Package créé: azure_deploy_all/"
echo "   - Taille totale: $SIZE"
echo "   - 6 modèles inclus (B0/B1/B2 × FP32/INT8)"
echo ""
echo "⚠️  ATTENTION: Package de ~220MB"
echo "   Azure F1 gratuit a une limite de 1GB"
echo "   Le déploiement peut prendre du temps"
