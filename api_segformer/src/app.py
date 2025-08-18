# src/app.py - Version complète Azure-ready
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

# Essayer d'importer ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime non disponible")

# Configuration selon l'environnement
IS_AZURE = os.environ.get('WEBSITE_SITE_NAME') is not None

if IS_AZURE:
    # Configuration Azure minimaliste
    from config_azure import Config
else:
    # Configuration locale complète
    from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Variables globales
model_session = None
current_model = Config.DEFAULT_MODEL
model_loaded = False

def preprocess_image_simple(image):
    """Prétraitement simple sans transformers pour Azure"""
    image = image.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT), Image.BILINEAR)
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_transformers(image):
    """Prétraitement avec transformers pour local"""
    try:
        from transformers import SegformerFeatureExtractor
        model_config = Config.MODELS[current_model]
        feature_extractor = SegformerFeatureExtractor.from_pretrained(
            model_config['feature_extractor'],
            size={"height": Config.IMG_HEIGHT, "width": Config.IMG_WIDTH}
        )
        inputs = feature_extractor(images=[image], return_tensors="np")
        return inputs['pixel_values']
    except:
        logger.warning("Transformers non disponible, utilisation du preprocessing simple")
        return preprocess_image_simple(image)

def ensure_model_loaded():
    """Charge le modèle si nécessaire (lazy loading)"""
    global model_session, model_loaded
    
    if not model_loaded:
        return init_model()
    return True

def init_model():
    """Initialise le modèle"""
    global model_session, model_loaded
    
    if not ONNX_AVAILABLE:
        logger.error("ONNX Runtime non disponible")
        return False
    
    model_config = Config.MODELS[current_model]
    logger.info(f"Chargement du modèle {current_model}...")
    
    model_path = model_config['path']
    
    # Sur Azure, utiliser FP32 si INT8 échoue
    if IS_AZURE and 'int8' in current_model:
        # Essayer d'abord INT8
        if not os.path.exists(model_path):
            # Fallback vers FP32
            fp32_path = model_path.replace('model_quantized.onnx', 'model.onnx')
            if os.path.exists(fp32_path):
                logger.info("Fallback vers modèle FP32")
                model_path = fp32_path
            else:
                logger.error(f"Aucun modèle trouvé")
                return False
    
    if not os.path.exists(model_path):
        logger.error(f"Fichier modèle non trouvé: {model_path}")
        return False
    
    try:
        # Essayer de charger le modèle
        providers = ['CPUExecutionProvider']
        model_session = ort.InferenceSession(model_path, providers=providers)
        model_loaded = True
        logger.info(f"✅ Modèle chargé: {model_config.get('size_mb', 'N/A')}MB, IoU: {model_config.get('iou', 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        
        # Sur Azure, essayer avec FP32 si INT8 échoue
        if IS_AZURE and 'int8' in current_model and 'quantized' in model_path:
            logger.info("Tentative avec modèle FP32...")
            fp32_path = model_path.replace('model_quantized.onnx', 'model.onnx')
            if os.path.exists(fp32_path):
                try:
                    model_session = ort.InferenceSession(fp32_path, providers=providers)
                    model_loaded = True
                    logger.info(f"✅ Modèle FP32 chargé en fallback")
                    return True
                except Exception as e2:
                    logger.error(f"Échec FP32 aussi: {e2}")
        
        return False

def apply_color_palette(mask):
    """Applique la palette de couleurs Cityscapes"""
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(Config.CITYSCAPES_COLORS):
        colored_mask[mask == class_id] = color
    return colored_mask

@app.route('/')
def index():
    """Page d'accueil avec interface et sélecteur de modèles"""
    model_info = Config.MODELS.get(current_model, {})
    model_name = current_model.replace('_', '-').upper().replace('SEGFORMER-', 'SegFormer-').replace('-INT8', ' INT8').replace('-FP32', ' FP32')
    
    # Liste des modèles pour le sélecteur
    models_options = []
    for model_id, info in Config.MODELS.items():
        if IS_AZURE:
            # Sur Azure, montrer seulement les modèles disponibles
            if os.path.exists(info['path']) or os.path.exists(info['path'].replace('model_quantized.onnx', 'model.onnx')):
                models_options.append({
                    'id': model_id,
                    'name': model_id.replace('_', '-').upper().replace('SEGFORMER-', 'SegFormer-'),
                    'info': f"{info.get('size_mb', 'N/A')}MB - IoU: {info.get('iou', 'N/A')}"
                })
        else:
            models_options.append({
                'id': model_id,
                'name': model_id.replace('_', '-').upper().replace('SEGFORMER-', 'SegFormer-'),
                'info': f"{info.get('size_mb', 'N/A')}MB - IoU: {info.get('iou', 'N/A')}"
            })
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>API SegFormer P9</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            .model-selector {
                text-align: center;
                margin: 20px 0;
                padding: 20px;
                background: #f9f9f9;
                border-radius: 8px;
            }
            .model-selector select {
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                border: 2px solid #764ba2;
                background: white;
                cursor: pointer;
                max-width: 500px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            .stat-card {
                background: #f7f7f7;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-card.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .upload-area { 
                border: 2px dashed #764ba2; 
                padding: 40px; 
                text-align: center;
                border-radius: 8px;
                margin: 20px 0;
            }
            button {
                background: #764ba2;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
            }
            button:hover { background: #5a3785; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .results { margin-top: 20px; }
            img { max-width: 100%; border-radius: 8px; margin: 10px 0; }
            .error { color: red; padding: 10px; background: #ffe0e0; border-radius: 5px; }
            .success { color: green; padding: 10px; background: #e0ffe0; border-radius: 5px; }
            .loading { text-align: center; padding: 20px; }
            .azure-warning {
                background: #fff3cd;
                border: 1px solid #ffc107;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 API SegFormer - Projet 9</h1>
            
            {% if is_azure %}
            <div class="azure-warning">
                ⚠️ Mode Azure F1 : Ressources limitées. Les modèles INT8 utilisent automatiquement FP32 en fallback.
            </div>
            {% endif %}
            
            <!-- Sélecteur de modèle -->
            <div class="model-selector">
                <label for="modelSelector"><strong>Choisir le modèle :</strong></label>
                <select id="modelSelector" onchange="changeModel()">
                    {% for model in models_options %}
                    <option value="{{ model.id }}" {% if model.id == current_model %}selected{% endif %}>
                        {{ model.name }} - {{ model.info }}
                    </option>
                    {% endfor %}
                </select>
                <div id="changeStatus"></div>
            </div>
            
            <div class="stats">
                <div class="stat-card active">
                    <h3>Modèle actuel</h3>
                    <p id="modelName">{{ model_name }}</p>
                </div>
                <div class="stat-card">
                    <h3>Performance</h3>
                    <p id="modelIoU">IoU: {{ model_iou }}</p>
                </div>
                <div class="stat-card">
                    <h3>Taille</h3>
                    <p id="modelSize">{{ model_size }} MB</p>
                </div>
            </div>
            
            <div class="upload-area">
                <h3>Testez la segmentation</h3>
                <input type="file" id="fileInput" accept="image/*">
                <button id="segmentBtn" onclick="uploadImage()">Segmenter l'image</button>
            </div>
            
            <div id="results" class="results"></div>
        </div>
        
        <script>
            // Changer de modèle
            async function changeModel() {
                const modelId = document.getElementById('modelSelector').value;
                const statusDiv = document.getElementById('changeStatus');
                
                statusDiv.innerHTML = '<div class="loading">Chargement du modèle...</div>';
                
                try {
                    const response = await fetch(`/switch/${modelId}`, {method: 'POST'});
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        statusDiv.innerHTML = '<div class="success">✅ Modèle changé avec succès</div>';
                        
                        // Mettre à jour l'affichage
                        document.getElementById('modelName').textContent = data.display_name;
                        document.getElementById('modelIoU').textContent = `IoU: ${data.info.iou}`;
                        document.getElementById('modelSize').textContent = `${data.info.size_mb} MB`;
                        
                        setTimeout(() => {
                            statusDiv.innerHTML = '';
                        }, 3000);
                    } else {
                        statusDiv.innerHTML = `<div class="error">❌ Erreur : ${data.error}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="error">❌ Erreur de connexion</div>`;
                }
            }
            
            // Upload et segmentation
            function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Veuillez sélectionner une image');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', file);
                
                const btn = document.getElementById('segmentBtn');
                btn.disabled = true;
                document.getElementById('results').innerHTML = '<div class="loading">Traitement en cours...</div>';
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    btn.disabled = false;
                    if (data.status === 'success') {
                        document.getElementById('results').innerHTML = `
                            <h3>Résultats</h3>
                            <p><strong>Temps d'inférence:</strong> ${data.inference_time_ms.toFixed(1)} ms</p>
                            <p><strong>Modèle utilisé:</strong> ${data.model}</p>
                            <h4>Segmentation:</h4>
                            <img src="data:image/png;base64,${data.segmentation_image}" alt="Segmentation">
                            <h4>Distribution des classes:</h4>
                            <pre>${JSON.stringify(data.class_distribution, null, 2)}</pre>
                        `;
                    } else {
                        document.getElementById('results').innerHTML = `<div class="error">Erreur: ${data.error}</div>`;
                    }
                })
                .catch(error => {
                    btn.disabled = false;
                    document.getElementById('results').innerHTML = `<div class="error">Erreur: ${error}</div>`;
                });
            }
        </script>
    </body>
    </html>
    ''', 
    is_azure=IS_AZURE,
    models_options=models_options,
    current_model=current_model,
    model_name=model_name,
    model_iou=model_info.get('iou', 'N/A'),
    model_size=model_info.get('size_mb', 'N/A'))


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de prédiction"""
    global model_session, current_model, model_loaded
    
    try:
        # NOUVEAU : Lire le paramètre model depuis la requête
        requested_model = request.args.get('model', current_model)
        
        # Si le modèle demandé est différent du modèle actuel, le charger
        if requested_model != current_model:
            logger.info(f"Changement de modèle: {current_model} -> {requested_model}")
            
            # Vérifier que le modèle existe
            if requested_model not in Config.MODELS:
                return jsonify({
                    'status': 'error', 
                    'error': f'Modèle {requested_model} non disponible'
                }), 400
            
            # Décharger l'ancien modèle
            model_session = None
            model_loaded = False
            
            # Changer le modèle courant
            current_model = requested_model
            
            # Charger le nouveau modèle
            if not init_model():
                return jsonify({
                    'status': 'error', 
                    'error': f'Impossible de charger le modèle {requested_model}'
                }), 500
        
        # Lazy loading si nécessaire
        if not ensure_model_loaded():
            return jsonify({'status': 'error', 'error': 'Impossible de charger le modèle'}), 500
        
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'error': 'Aucune image fournie'}), 400
        
        # Charger l'image
        image = Image.open(request.files['image'].stream).convert('RGB')
        original_size = image.size
        
        # Prétraitement selon l'environnement
        if IS_AZURE:
            img_array = preprocess_image_simple(image)
        else:
            img_array = preprocess_image_transformers(image)
        
        # Inférence
        start_time = time.time()
        outputs = model_session.run(None, {model_session.get_inputs()[0].name: img_array})
        inference_time = (time.time() - start_time) * 1000
        
        # Post-traitement
        logits = outputs[0][0]  # [C, H, W]
        
        # Upsampling si nécessaire
        if logits.shape[1] != Config.IMG_HEIGHT or logits.shape[2] != Config.IMG_WIDTH:
            from PIL import Image as PILImage
            pred_mask = np.argmax(logits, axis=0)
            mask_pil = PILImage.fromarray(pred_mask.astype(np.uint8))
            mask_pil = mask_pil.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT), PILImage.NEAREST)
            pred_mask = np.array(mask_pil)
        else:
            pred_mask = np.argmax(logits, axis=0)
        
        # Appliquer les couleurs
        colored_mask = apply_color_palette(pred_mask)
        
        # Convertir en base64
        mask_pil = Image.fromarray(colored_mask)
        buffer = io.BytesIO()
        mask_pil.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Statistiques
        unique, counts = np.unique(pred_mask, return_counts=True)
        class_distribution = {
            Config.CITYSCAPES_CLASSES[i]: float(counts[list(unique).index(i)] / pred_mask.size * 100)
            for i in unique if i < len(Config.CITYSCAPES_CLASSES)
        }
        
        return jsonify({
            'status': 'success',
            'model': current_model,  # Retourner le modèle réellement utilisé
            'inference_time_ms': inference_time,
            'class_distribution': class_distribution,
            'segmentation_image': mask_base64
        })
        
    except Exception as e:
        logger.error(f"Erreur dans predict: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/switch/<model_id>', methods=['POST'])
def switch_model(model_id):
    """Change le modèle actif"""
    global model_session, current_model, model_loaded
    
    if model_id not in Config.MODELS:
        return jsonify({'status': 'error', 'error': f'Modèle {model_id} non disponible'}), 400
    
    try:
        # Décharger l'ancien modèle
        model_session = None
        model_loaded = False
        
        # Changer le modèle courant
        current_model = model_id
        
        # Charger le nouveau modèle
        if init_model():
            model_info = Config.MODELS[model_id]
            display_name = model_id.replace('_', '-').upper().replace('SEGFORMER-', 'SegFormer-').replace('-INT8', ' INT8').replace('-FP32', ' FP32')
            
            return jsonify({
                'status': 'success',
                'model': current_model,
                'display_name': display_name,
                'info': model_info
            })
        else:
            return jsonify({'status': 'error', 'error': 'Impossible de charger le modèle'}), 500
            
    except Exception as e:
        logger.error(f"Erreur lors du changement de modèle: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """Liste tous les modèles disponibles"""
    models_list = []
    for model_id, info in Config.MODELS.items():
        # Vérifier si le modèle existe
        available = os.path.exists(info['path']) or (IS_AZURE and os.path.exists(info['path'].replace('model_quantized.onnx', 'model.onnx')))
        
        models_list.append({
            'id': model_id,
            'name': model_id.replace('_', '-').upper(),
            'precision': info.get('precision', 'FP32'),
            'size_mb': info.get('size_mb', 'N/A'),
            'iou': info.get('iou', 'N/A'),
            'available': available,
            'active': model_id == current_model
        })
    
    return jsonify({
        'models': models_list,
        'current': current_model,
        'is_azure': IS_AZURE
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check pour Azure"""
    return jsonify({
        'status': 'healthy',
        'model': current_model,
        'model_loaded': model_loaded,
        'is_azure': IS_AZURE,
        'timestamp': time.time()
    })

@app.route('/test_onnx')
def test_onnx():
    """Test de diagnostic ONNX directement intégré"""
    import onnxruntime as ort
    
    output = []
    output.append("🔍 Diagnostic ONNX sur Azure")
    output.append(f"Python : {sys.version}")
    output.append(f"ONNX Runtime : {ort.__version__}")
    output.append(f"Providers : {ort.get_available_providers()}")
    output.append("")
    
    models_to_test = {
        'B1 INT8': 'models/segformer_b1/model_quantized.onnx',
        'B1 FP32': 'models/segformer_b1/model.onnx',
        'B0 INT8': 'models/segformer_b0/model_quantized.onnx',
        'B0 FP32': 'models/segformer_b0/model.onnx',
    }
    
    results = {}
    
    for name, model_path in models_to_test.items():
        output.append(f"\n{'='*50}")
        output.append(f"Test : {name} ({model_path})")
        output.append(f"{'='*50}")
        
        if not os.path.exists(model_path):
            output.append(f"❌ Fichier non trouvé")
            results[name] = "Non trouvé"
            continue
        
        # Taille du fichier
        size_mb = os.path.getsize(model_path) / (1024*1024)
        output.append(f"📊 Taille : {size_mb:.2f} MB")
        
        # Essayer de charger avec ONNX pour voir les opsets
        try:
            import onnx
            model = onnx.load(model_path)
            for opset in model.opset_import:
                output.append(f"   Opset : {opset.domain or 'ai.onnx'} version {opset.version}")
        except ImportError:
            output.append("   ⚠️ Package onnx non disponible")
        except Exception as e:
            output.append(f"   ⚠️ Erreur lecture ONNX : {str(e)[:100]}")
        
        # Test ONNX Runtime
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            output.append(f"✅ ONNX Runtime : Chargement réussi!")
            
            # Info inputs/outputs
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            output.append(f"   Inputs : {inputs[0].name} {inputs[0].shape}")
            output.append(f"   Outputs : {outputs[0].name} {outputs[0].shape}")
            
            # Test inférence
            dummy_input = np.random.randn(1, 3, 256, 512).astype(np.float32)
            result = session.run(None, {inputs[0].name: dummy_input})
            output.append(f"✅ Inférence réussie! Shape: {result[0].shape}")
            results[name] = "✅ OK"
            
        except Exception as e:
            output.append(f"❌ Erreur ONNX Runtime :")
            error_msg = str(e)
            if "opset" in error_msg.lower():
                # Extraire l'info sur l'opset
                output.append(f"   Problème d'opset détecté")
            output.append(f"   {error_msg[:200]}")
            results[name] = "❌ Erreur"
    
    output.append(f"\n{'='*50}")
    output.append("RÉSUMÉ")
    output.append(f"{'='*50}")
    for name, status in results.items():
        output.append(f"{status} - {name}")
    
    # Retourner le résultat formaté
    return f"<pre>{'<br>'.join(output)}</pre>"

@app.route('/convert_opset', methods=['POST'])
def convert_opset():
    """Convertit les modèles INT8 à un opset compatible"""
    output = []
    output.append("🔄 Conversion des modèles INT8...")
    
    try:
        import onnx
        from onnx import version_converter
        
        models = ['segformer_b1']  # Commencer par B1
        
        for model_name in models:
            input_path = f'models/{model_name}/model_quantized.onnx'
            output_path = f'models/{model_name}/model_quantized_opset11.onnx'
            
            if not os.path.exists(input_path):
                output.append(f"❌ {model_name} : fichier non trouvé")
                continue
            
            try:
                # Charger le modèle
                model = onnx.load(input_path)
                
                # Convertir à opset 11
                converted = version_converter.convert_version(model, 11)
                
                # Sauvegarder
                onnx.save(converted, output_path)
                
                # Tester
                session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
                output.append(f"✅ {model_name} : converti à opset 11 et testé")
                
            except Exception as e:
                output.append(f"❌ {model_name} : {str(e)[:100]}")
        
    except ImportError:
        output.append("❌ Package onnx non installé")
        output.append("Ajoutez 'onnx' dans requirements.txt")
    
    return jsonify({'status': 'done', 'output': output})

if __name__ == '__main__':
    # Ne pas précharger le modèle (lazy loading)
    port = Config.PORT
    logger.info(f"✅ API prête sur le port {port}")
    logger.info(f"📦 Modèle par défaut : {Config.DEFAULT_MODEL}")
    logger.info(f"☁️  Mode Azure : {IS_AZURE}")
    
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)