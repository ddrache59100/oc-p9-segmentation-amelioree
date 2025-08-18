#!/bin/bash
# prepare_minimal_deployment.sh - Version corrigée

MODE=${1:-b1-only}

echo "📦 Préparation du déploiement Azure pour SegFormer P9..."
echo "   Mode: $MODE"

# Nettoyer et créer le dossier
rm -rf azure_deploy_minimal
mkdir -p azure_deploy_minimal

# Copier l'application SANS le dossier src (structure plate)
echo "📄 Création de application.py..."
cat > azure_deploy_minimal/application.py << 'EOF'
import os
import sys
import time
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import io
import base64
import onnxruntime as ort

# Configuration directe sans import
IMG_HEIGHT = 256
IMG_WIDTH = 512
NUM_CLASSES = 8

CITYSCAPES_CLASSES = ['flat', 'human', 'vehicle', 'construction', 
                      'object', 'nature', 'sky', 'void']

CITYSCAPES_COLORS = [
    [128, 64, 128], [220, 20, 60], [0, 0, 142], [190, 153, 153],
    [153, 153, 153], [107, 142, 35], [70, 130, 180], [0, 0, 0]
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Variables globales
model_session = None
model_loaded = False

def init_model():
    global model_session, model_loaded
    
    logger.info("Chargement du modèle SegFormer-B1 INT8...")
    
    model_path = 'models/segformer_b1/model_quantized.onnx'
    if not os.path.exists(model_path):
        logger.error(f"Modèle non trouvé: {model_path}")
        return False
    
    try:
        model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        model_loaded = True
        logger.info("✅ Modèle chargé (14.2 MB)")
        return True
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        return False

def apply_color_palette(mask):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_COLORS):
        colored_mask[mask == class_id] = color
    return colored_mask

def preprocess_image(image):
    """Prétraitement simple sans transformers"""
    # Resize
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    # Convert to array
    img_array = np.array(image).astype(np.float32)
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    # Transpose to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>API SegFormer P9</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
            .stat-card { background: #f0f0f0; padding: 15px; text-align: center; border-radius: 8px; }
            .upload-area { border: 2px dashed #333; padding: 40px; text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; }
            img { max-width: 100%; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🚗 API SegFormer - Projet 9</h1>
        <div class="stats">
            <div class="stat-card">
                <h3>Modèle</h3>
                <p>SegFormer-B1 INT8</p>
            </div>
            <div class="stat-card">
                <h3>Performance</h3>
                <p>IoU: 0.667</p>
            </div>
            <div class="stat-card">
                <h3>Taille</h3>
                <p>14.2 MB</p>
            </div>
        </div>
        <div class="upload-area">
            <input type="file" id="fileInput" accept="image/*">
            <button onclick="uploadImage()">Segmenter</button>
        </div>
        <div id="results"></div>
        <script>
            function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('image', file);
                
                document.getElementById('results').innerHTML = '<p>Traitement en cours...</p>';
                
                fetch('/predict', {method: 'POST', body: formData})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('results').innerHTML = 
                            '<h3>Résultats</h3>' +
                            '<p>Temps: ' + data.inference_time_ms.toFixed(1) + ' ms</p>' +
                            '<img src="data:image/png;base64,' + data.segmentation_image + '">';
                    } else {
                        document.getElementById('results').innerHTML = '<p>Erreur: ' + data.error + '</p>';
                    }
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lazy loading
        if not model_loaded:
            if not init_model():
                return jsonify({'status': 'error', 'error': 'Modèle non chargé'}), 500
        
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'error': 'Aucune image'}), 400
        
        # Charger l'image
        image = Image.open(request.files['image'].stream).convert('RGB')
        
        # Prétraitement simple
        img_array = preprocess_image(image)
        
        # Inférence
        start_time = time.time()
        outputs = model_session.run(None, {model_session.get_inputs()[0].name: img_array})
        inference_time = (time.time() - start_time) * 1000
        
        # Post-traitement
        logits = outputs[0][0]
        pred_mask = np.argmax(logits, axis=0)
        
        # Appliquer les couleurs
        colored_mask = apply_color_palette(pred_mask)
        
        # Convertir en base64
        mask_pil = Image.fromarray(colored_mask)
        buffer = io.BytesIO()
        mask_pil.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'status': 'success',
            'inference_time_ms': inference_time,
            'segmentation_image': mask_base64
        })
        
    except Exception as e:
        logger.error(f"Erreur predict: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"API démarrage sur port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
EOF

# Copier le modèle
echo "🤖 Copie du modèle..."
mkdir -p azure_deploy_minimal/models/segformer_b1
cp models/segformer_b1/model_quantized.onnx azure_deploy_minimal/models/segformer_b1/

# Requirements minimal
echo "📝 Requirements minimal..."
cat > azure_deploy_minimal/requirements.txt << EOF
Flask==2.3.2
flask-cors==4.0.0
Pillow==10.0.0
numpy==1.24.3
onnxruntime==1.16.0
gunicorn==21.2.0
EOF

# Startup - IMPORTANT: référencer application:app
echo "🚀 Configuration startup..."
echo "gunicorn --bind=0.0.0.0 --timeout 600 application:app" > azure_deploy_minimal/startup.txt

# Fichiers Azure
echo "python-3.9" > azure_deploy_minimal/runtime.txt

# Calculer la taille
SIZE=$(du -sh azure_deploy_minimal | cut -f1)
echo ""
echo "✅ Package créé: azure_deploy_minimal/"
echo "   - Taille: $SIZE"
echo "   - Structure plate (pas de dossier src)"
echo "   - Sans transformers ni torch"
echo "   - Modèle: SegFormer-B1 INT8"