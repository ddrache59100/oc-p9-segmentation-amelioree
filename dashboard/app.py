"""
Application Streamlit Comparative - P8 vs P9
Comparaison des modèles de segmentation
VGG16-UNet (P8) vs SegFormer (P9)
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import numpy as np
import time
from pathlib import Path
import os
import pandas as pd
import altair as alt
from streamlit_image_comparison import image_comparison


# Classes Cityscapes
CITYSCAPES_CLASSES = ['flat', 'human', 'vehicle', 'construction', 
                      'object', 'nature', 'sky', 'void']

# URLs des APIs
API_URLS = {
    'P8': os.environ.get('API_URL_P8', 'https://oc-p8-segmentation.azurewebsites.net'),
    'P9': os.environ.get('API_URL_P9', 'https://oc-p9-segformer.azurewebsites.net')
}

# Configuration des modèles
MODELS_CONFIG = {
    'P8': {
        'VGG16-UNet': {
            'endpoint': '/predict/visualize',
            'size_mb': 98.8,
            'iou': 0.631,
            'iou_weighted': 0.770,
            'architecture': 'CNN Encoder-Decoder',
            'technique': 'Architecture classique'
        }
    },
    'P9': {
        'SegFormer-B0 FP32': {
            'model': 'segformer_b0_fp32',
            'size_mb': 14.5,
            'iou': 0.698,
            'architecture': 'Vision Transformer',
            'technique': 'Hierarchical Transformer'
        },
        'SegFormer-B0 INT8': {
            'model': 'segformer_b0_int8',
            'size_mb': 4.6,
            'iou': 0.587,
            'architecture': 'Vision Transformer',
            'technique': 'Quantification INT8'
        },
        'SegFormer-B1 FP32': {
            'model': 'segformer_b1_fp32',
            'size_mb': 52.5,
            'iou': 0.701,
            'architecture': 'Vision Transformer',
            'technique': 'Hierarchical Transformer'
        },
        'SegFormer-B1 INT8': {
            'model': 'segformer_b1_int8',
            'size_mb': 14.2,
            'iou': 0.667,
            'architecture': 'Vision Transformer',
            'technique': 'Quantification INT8'
        },
        'SegFormer-B2 FP32': {
            'model': 'segformer_b2_fp32',
            'size_mb': 104.9,
            'iou': 0.760,
            'architecture': 'Vision Transformer',
            'technique': 'Hierarchical Transformer'
        },
        'SegFormer-B2 INT8': {
            'model': 'segformer_b2_int8',
            'size_mb': 28.3,
            'iou': 0.705,
            'architecture': 'Vision Transformer',
            'technique': 'Quantification INT8'
        }
    }
}

# Chemins
IMAGES_PATH = Path("test_images")

# Palette OFFICIELLE Cityscapes en HEX
COLOR_MAPPING = {
    'flat': '#804080',      # [128, 64, 128] - Violet (route/trottoir)
    'human': '#DC143C',     # [220, 20, 60] - Rouge (personne)
    'vehicle': '#00008E',   # [0, 0, 142] - Bleu foncé (véhicule)
    'construction': '#464646', # [70, 70, 70] - Gris foncé (bâtiment) ← CORRIGÉ !
    'object': '#999999',    # [153, 153, 153] - Gris clair (objet)
    'nature': '#6B8E23',    # [107, 142, 35] - Vert olive (végétation)
    'sky': '#4682B4',       # [70, 130, 180] - Bleu ciel
    'void': '#000000'       # [0, 0, 0] - Noir (non classifié)
}


#  Configuration de la page
st.set_page_config(
    page_title="Comparaison Segmentation P8 vs P9 - Future Vision Transport",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés (version simplifiée)
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
    }
    .element-container:has(h1) {
        margin-top: 0;
        padding-top: 0;
    }
    header[data-testid="stHeader"] {
        height: 0rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        padding: 10px;
        border-radius: 5px;
    }
    .stImage {
        border: 1px solid #e0e2e6;
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] > div {
        height: 100%;
        padding-top: 0;
    }
    section[data-testid="stSidebar"] .element-container:first-child {
        margin-top: -1rem;
    }
    .comparison-container {
        border: 2px solid #e0e2e6;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    /* Réduire l'espace des checkboxes */
    section[data-testid="stSidebar"] .row-widget.stCheckbox {
        margin-bottom: -0.5rem !important;
        margin-top: -0.5rem !important;
    }
    .winner-badge {
        background-color: #28a745;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    /* Forcer le composant image_comparison à respecter la largeur de sa colonne */
    .stColumn iframe[title*="streamlit_image_comparison"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Ajuster spécifiquement dans col_swipe (première colonne) */
    .stColumn:first-child iframe[title*="streamlit_image_comparison"] {
        width: 100% !important;
        height: auto !important;
    }
    
    /* S'assurer que les images suivent aussi */
    .stColumn:first-child .stImage {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Fonction utilitaires reprises du P8
# @st.cache_data
# def load_image_index():
#     """Charge l'index des images disponibles"""
#     index_path = IMAGES_PATH / "index.json"
#     if index_path.exists():
#         with open(index_path, 'r') as f:
#             return json.load(f)
#     return {"images": [], "description": "Aucune image trouvée"}

@st.cache_data
def load_image_index():
    """Charge l'index des images disponibles (le génère si nécessaire)"""
    # Générer l'index si nécessaire
    generate_image_index()
    
    # Charger l'index
    index_path = IMAGES_PATH / "index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de l'index : {e}")
    
    # Fallback : retourner un index vide
    return {"images": [], "description": "Aucune image trouvée"}

@st.cache_data
def load_image(image_path):
    """Charge une image depuis le disque"""
    return Image.open(image_path)

def image_to_base64(image):
    """Convertit une image PIL en base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base64_to_image(base64_string):
    """Convertit une string base64 en image PIL"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def check_api_health(api_name):
    """Vérifie l'état d'une API"""
    try:
        response = requests.get(f"{API_URLS[api_name]}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def predict_p8(image_base64):
    """Appelle l'API P8 pour prédire la segmentation"""
    try:
        response = requests.post(
            f"{API_URLS['P8']}/predict/visualize",
            json={"image": image_base64},
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            
            # Adapter les clés P8 pour uniformiser
            if 'colored_mask' in result:
                result['mask'] = result['colored_mask']
            if 'processing_time_ms' in result:
                result['inference_time'] = result['processing_time_ms']
            
            # Garder aussi l'overlay si disponible
            if 'overlay' in result:
                result['overlay_image'] = result['overlay']
                
            return True, result
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def predict_p9(image_file, model_name):
    """Appelle l'API P9 pour prédire la segmentation"""
    try:
        files = {'image': ('image.jpg', image_file, 'image/jpeg')}
        params = {'model': MODELS_CONFIG['P9'][model_name]['model']}
        
        response = requests.post(
            f"{API_URLS['P9']}/predict",
            files=files,
            params=params,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Adapter les clés P9 pour uniformiser
            if 'segmentation_image' in result:
                result['mask'] = result['segmentation_image']
            if 'inference_time_ms' in result:
                result['inference_time'] = result['inference_time_ms']
                
            return True, result
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def generate_image_index():
    """Génère l'index des images si nécessaire"""
    index_path = IMAGES_PATH / "index.json"
    
    # Si l'index existe et est récent, ne pas le régénérer
    if index_path.exists():
        # Vérifier si l'index est plus récent que les images
        index_mtime = index_path.stat().st_mtime
        images_mtime = max((f.stat().st_mtime for f in IMAGES_PATH.glob("*.png") 
                           if not f.name.endswith("_mask.png")), default=0)
        
        if index_mtime >= images_mtime:
            # L'index est à jour
            return
    
    # Générer l'index
    image_files = sorted([f for f in IMAGES_PATH.glob("*.png") 
                         if not f.name.endswith("_mask.png")])
    
    images_data = []
    for img_file in image_files:
        img_id = img_file.stem  # "test_000", "test_001", etc.
        
        images_data.append({
            "id": img_id,
            "name": f"Image {img_id.replace('test_', '')}",
            "filename": img_file.name,
            "description": f"Image de test {img_id.replace('test_', '')}",
            "has_mask": (IMAGES_PATH / f"{img_id}_mask.png").exists()
        })
    
    index = {
        "description": "Images de test Cityscapes pour segmentation",
        "images": images_data,
        "total": len(images_data)
    }
    
    # Sauvegarder
    try:
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"✅ Index généré/mis à jour avec {len(images_data)} images")
    except Exception as e:
        print(f"⚠️ Erreur lors de la génération de l'index : {e}")
    
    return index

def create_overlay(original_image, mask_image, alpha=0.5):
    """Crée un overlay du masque sur l'image originale"""
    # Convertir en RGBA
    original = original_image.convert("RGBA")
    mask = mask_image.convert("RGBA")
    
    # Redimensionner le masque si nécessaire
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)
    
    # Créer l'overlay avec transparence
    overlay = Image.blend(original, mask, alpha)
    return overlay

# MODIFICATION 1 : Ajouter les métriques sur le masque prédit
def add_metrics_overlay(image, model_name, iou, time_ms, size_mb):
    """Ajoute les métriques directement sur l'image du masque"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Convertir en PIL si nécessaire
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    
    # Texte à afficher
    text_lines = [
        f"Modèle: {model_name}",
        f"IoU: {iou:.3f}",
        f"Temps: {time_ms:.0f}ms",
        f"Taille: {size_mb:.1f}MB"
    ]
    
    # Position (en bas à droite)
    x = img.width - 200
    y = img.height - 100
    
    # Rectangle semi-transparent pour le fond
    draw.rectangle(
        [(x-10, y-10), (img.width-10, img.height-10)],
        fill=(0, 0, 0, 180)  # Noir semi-transparent
    )
    
    # Texte en blanc
    for i, line in enumerate(text_lines):
        draw.text(
            (x, y + i*20),
            line,
            fill=(255, 255, 255),
            font=ImageFont.load_default()
        )
    
    return img

def create_distribution_chart(class_distribution):
    """Crée un graphique de distribution des classes avec Altair"""
    if not class_distribution:
        return None
    
    # Préparer les données en gérant les deux formats
    data = []
    for class_name, value in class_distribution.items():
        # Gérer les deux formats possibles
        if isinstance(value, dict):
            # Format P8: {'percentage': 12.5, ...}
            percentage = value.get('percentage', 0)
        else:
            # Format P9: directement un float
            percentage = float(value)
        
        data.append({
            'Classe': class_name,
            'Pourcentage': percentage
        })
    
    df = pd.DataFrame(data)
    if df.empty:
        return None
        
    df = df.sort_values('Pourcentage', ascending=False)
    
    # Créer le graphique
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Classe:N', 
            axis=alt.Axis(labelAngle=-45, labelFontSize=10, title=None)
        ),
        y=alt.Y('Pourcentage:Q',
            axis=alt.Axis(labelFontSize=10, title='%')
        ),
        color=alt.Color('Classe:N', 
            scale=alt.Scale(
                domain=list(COLOR_MAPPING.keys()),
                range=list(COLOR_MAPPING.values())
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('Classe:N', title='Classe'),
            alt.Tooltip('Pourcentage:Q', format='.1f', title='Pourcentage (%)')
        ]
    ).properties(
        height=180,
        title=alt.TitleParams(
            text='Distribution des classes',
            fontSize=12,
            anchor='start',
            offset=5
        )
    ).configure_view(strokeWidth=0).configure_axis(grid=False)
    
    return chart


def create_comparison_chart(df):
    # Ajouter une colonne Type
    df['Type'] = df['Modèle'].apply(
        lambda x: 'INT8' if 'INT8' in x else 'FP32'
    )
    
    chart = alt.Chart(df).mark_point(filled=True).encode(
        x=alt.X('Temps (ms):Q', 
            scale=alt.Scale(type='log'),
            title='Temps d\'inférence (ms) - échelle log'
        ),
        y=alt.Y('IoU:Q', 
            scale=alt.Scale(domain=[0.5, 0.8]),
            title='IoU Score'
        ),
        size=alt.Size('Taille (MB):Q', 
            scale=alt.Scale(range=[100, 1000]),
            legend=alt.Legend(title="Taille (MB)")
        ),
        color=alt.Color('Projet:N',
            scale=alt.Scale(
                domain=['P8', 'P9'],
                range=['#e74c3c', '#3498db']
            ),
            legend=alt.Legend(title="Projet")
        ),
        shape=alt.Shape('Type:N',
            scale=alt.Scale(
                domain=['FP32', 'INT8'],
                range=['circle', 'diamond']
            ),
            legend=alt.Legend(title="Précision")
        ),
        tooltip=[
            alt.Tooltip('Modèle:N', title='Modèle'),
            alt.Tooltip('IoU:Q', format='.3f'),
            alt.Tooltip('Temps (ms):Q', format='.0f'),
            alt.Tooltip('Taille (MB):Q', format='.1f'),
            alt.Tooltip('Type:N', title='Format')
        ]
    ).properties(
        width=700,
        height=400,
        title={
            "text": "Performance vs Efficacité",
            "subtitle": "Position idéale : haut-gauche (haute précision, faible latence)"
        }
    ).interactive()
    
    return chart


###### ajout IoU
def calculate_iou_metrics(pred_mask, gt_mask, debug=False):
    """
    Calcule l'IoU par classe et global entre masque prédit et ground truth
    
    Args:
        pred_mask: masque prédit (numpy array ou PIL Image)
        gt_mask: masque ground truth (numpy array ou PIL Image)
        debug: affiche des informations de debug
    
    Returns:
        dict avec IoU par classe, IoU global, IoU pondéré
    """
    # Convertir en numpy array si nécessaire
    if isinstance(pred_mask, Image.Image):
        pred_mask = np.array(pred_mask)
    if isinstance(gt_mask, Image.Image):
        gt_mask = np.array(gt_mask)
    
    # Debug : afficher les shapes et types
    if debug:
        print(f"Pred mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
        print(f"GT mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
        print(f"Pred unique values: {np.unique(pred_mask)[:20]}")  # Premiers 20 valeurs uniques
        print(f"GT unique values: {np.unique(gt_mask)[:20]}")
    
    # S'assurer que les masques sont en 2D (un seul canal)
    if len(pred_mask.shape) == 3:
        # Si c'est une image RGB, on peut avoir plusieurs cas :
        # 1. Les 3 canaux sont identiques (image en niveaux de gris répétée)
        if np.allclose(pred_mask[:,:,0], pred_mask[:,:,1]) and np.allclose(pred_mask[:,:,1], pred_mask[:,:,2]):
            pred_mask = pred_mask[:, :, 0]
        # 2. C'est une image colorée, il faut la convertir en indices
        else:
            # Convertir l'image RGB en indices de classes
            pred_mask = rgb_to_class_indices(pred_mask)
    
    if len(gt_mask.shape) == 3:
        if np.allclose(gt_mask[:,:,0], gt_mask[:,:,1]) and np.allclose(gt_mask[:,:,1], gt_mask[:,:,2]):
            gt_mask = gt_mask[:, :, 0]
        else:
            gt_mask = rgb_to_class_indices(gt_mask)
    
    # Vérifier que les valeurs sont dans la plage 0-7
    if pred_mask.max() > 7 or pred_mask.min() < 0:
        if debug:
            print(f"Warning: Pred mask values out of range [0-7]: min={pred_mask.min()}, max={pred_mask.max()}")
        # Essayer de normaliser si les valeurs sont dans une autre plage
        if pred_mask.max() > 100:  # Probablement des valeurs 0-255
            pred_mask = pred_mask // 32  # Diviser par 32 pour obtenir 0-7
    
    if gt_mask.max() > 7 or gt_mask.min() < 0:
        if debug:
            print(f"Warning: GT mask values out of range [0-7]: min={gt_mask.min()}, max={gt_mask.max()}")
        if gt_mask.max() > 100:
            gt_mask = gt_mask // 32
    
    # Redimensionner si nécessaire pour avoir la même taille
    if pred_mask.shape != gt_mask.shape:
        if debug:
            print(f"Resizing pred_mask from {pred_mask.shape} to {gt_mask.shape}")
        pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8))
        pred_mask_img = pred_mask_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
        pred_mask = np.array(pred_mask_img)
    
    # Calculer l'IoU par classe
    iou_per_class = {}
    pixels_per_class = {}
    
    for class_id in range(8):  # 8 classes Cityscapes
        # Masques binaires pour cette classe
        pred_class = (pred_mask == class_id)
        gt_class = (gt_mask == class_id)
        
        # Intersection et Union
        intersection = np.sum(pred_class & gt_class)
        union = np.sum(pred_class | gt_class)
        
        # Nombre de pixels GT pour cette classe (pour pondération)
        pixels_per_class[class_id] = np.sum(gt_class)
        
        # IoU pour cette classe
        if union > 0:
            iou = intersection / union
            iou_per_class[class_id] = iou
        else:
            # Pas de pixels de cette classe dans GT ni prédiction
            iou_per_class[class_id] = 1.0 if pixels_per_class[class_id] == 0 else 0.0
    
    # IoU global (moyenne simple)
    valid_ious = [iou for class_id, iou in iou_per_class.items() if pixels_per_class[class_id] > 0]
    iou_global = np.mean(valid_ious) if valid_ious else 0.0
    
    # IoU pondéré (pondéré par le nombre de pixels de chaque classe)
    total_pixels = sum(pixels_per_class.values())
    iou_weighted = 0.0
    if total_pixels > 0:
        for class_id, iou in iou_per_class.items():
            weight = pixels_per_class[class_id] / total_pixels
            iou_weighted += iou * weight
    
    return {
        'iou_per_class': iou_per_class,
        'iou_global': iou_global,
        'iou_weighted': iou_weighted,
        'pixels_per_class': pixels_per_class
    }

def rgb_to_class_indices(rgb_mask):
    """
    Convertit un masque RGB en indices de classes basé sur les couleurs Cityscapes
    """
    # Définition des couleurs Cityscapes (RGB)
    CITYSCAPES_COLORS_RGB = {
        0: [128, 64, 128],   # flat - violet
        1: [220, 20, 60],    # human - rouge
        2: [0, 0, 142],      # vehicle - bleu foncé
        3: [70, 70, 70],     # construction - gris foncé
        4: [153, 153, 153],  # object - gris clair
        5: [107, 142, 35],   # nature - vert
        6: [70, 130, 180],   # sky - bleu ciel
        7: [0, 0, 0]         # void - noir
    }
    
    h, w = rgb_mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Pour chaque classe, trouver les pixels correspondants
    for class_id, color in CITYSCAPES_COLORS_RGB.items():
        # Créer un masque pour cette couleur
        mask = np.all(rgb_mask == color, axis=2)
        class_mask[mask] = class_id
    
    return class_mask

def display_iou_metrics(iou_metrics, show_details=True):
    """
    Affiche les métriques IoU de manière formatée
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("IoU Global", f"{iou_metrics['iou_global']:.3f}",
                 help="Moyenne simple des IoU de toutes les classes présentes")
    
    with col2:
        st.metric("IoU Pondéré", f"{iou_metrics['iou_weighted']:.3f}",
                 help="IoU pondéré par le nombre de pixels de chaque classe")
    
    if show_details:
        with st.expander("📊 IoU détaillé par classe", expanded=False):
            # Créer un DataFrame pour l'affichage
            iou_data = []
            for class_id in range(8):
                class_name = CITYSCAPES_CLASSES[class_id] if class_id < len(CITYSCAPES_CLASSES) else f"Classe {class_id}"
                iou = iou_metrics['iou_per_class'].get(class_id, 0)
                pixels = iou_metrics['pixels_per_class'].get(class_id, 0)
                
                if pixels > 0:  # Seulement afficher les classes présentes
                    iou_data.append({
                        'Classe': class_name,
                        'IoU': iou,
                        'Pixels GT': pixels,
                        'Pourcentage': (pixels / sum(iou_metrics['pixels_per_class'].values()) * 100)
                    })
            
            if iou_data:
                df_iou = pd.DataFrame(iou_data)
                df_iou = df_iou.sort_values('IoU', ascending=False)
                
                # Formater l'affichage
                st.dataframe(
                    df_iou.style.format({
                        'IoU': '{:.3f}',
                        'Pixels GT': '{:,.0f}',
                        'Pourcentage': '{:.1f}%'
                    }).background_gradient(subset=['IoU'], cmap='RdYlGn', vmin=0, vmax=1),
                    use_container_width=True
                )

# Pour débugger et voir exactement quelle valeur est dans la cellule
def debug_cell_value(df, row_name, col_name):
    """Fonction pour débugger une valeur spécifique du tableau"""
    try:
        value = df.loc[row_name, col_name]
        print(f"Valeur brute : {value}")
        print(f"Type : {type(value)}")
        print(f"String : '{str(value)}'")
        print(f"Float : {float(value)}")
        print(f"Est proche de 0 ? {abs(float(value)) < 0.001}")
    except Exception as e:
        print(f"Erreur : {e}")


# Modifier aussi la fonction create_comparative_iou_table pour mieux gérer les NaN
def create_comparative_iou_table(comparison_results, mask_gt, debug=False):
    """
    Crée un tableau comparatif des métriques IoU pour tous les modèles
    
    Logique pour les IoU par classe :
    - NaN : classe absente du GT ET de la prédiction (cas correct, sera gris)
    - 0.0 : classe présente dans GT ou prédiction mais IoU=0 (erreur, sera rouge)
    - >0 : IoU normal avec code couleur selon valeur
    """
    import numpy as np
    import pandas as pd
    
    # DEBUG : Analyser le Ground Truth
    if debug:
        if mask_gt is not None:
            print("\n" + "="*60)
            print("DEBUG - Analyse du Ground Truth")
            print("="*60)
            
            # Convertir en array numpy si nécessaire
            if isinstance(mask_gt, Image.Image):
                mask_gt_np = np.array(mask_gt)
            else:
                mask_gt_np = mask_gt
                
            # Si RGB, prendre un seul canal
            if len(mask_gt_np.shape) == 3:
                mask_gt_np = mask_gt_np[:, :, 0]
            
            print(f"Shape du GT : {mask_gt_np.shape}")
            print(f"Valeurs uniques dans GT : {np.unique(mask_gt_np)}")
            
            # Compter les pixels par classe
            print("\nNombre de pixels par classe dans le GT :")
            print("-" * 40)
            total_pixels = mask_gt_np.size
            for class_id in range(8):
                pixel_count = np.sum(mask_gt_np == class_id)
                percentage = (pixel_count / total_pixels) * 100
                class_name = CITYSCAPES_CLASSES[class_id] if class_id < len(CITYSCAPES_CLASSES) else f"Classe {class_id}"
                
                if pixel_count > 0:
                    print(f"Classe {class_id} ({class_name:12s}) : {pixel_count:8d} pixels ({percentage:6.2f}%)")
                else:
                    print(f"Classe {class_id} ({class_name:12s}) : ABSENTE")
            
            print("-" * 40)
            print(f"Total : {total_pixels} pixels")
            print("="*60 + "\n")
        else:
            print("\n⚠️ Pas de Ground Truth disponible\n")
    
    # # Préparer les données pour le tableau
    # table_data = {
    #     'Métrique': [
    #         'IoU Modèle',
    #         'IoU Réel',
    #         'IoU Pondéré',
    #         'Temps (ms)',
    #         'Taille (MB)',
    #         '---',  # Séparateur
    #         'IoU flat',
    #         'IoU human',
    #         'IoU vehicle',
    #         'IoU construction',
    #         'IoU object',
    #         'IoU nature',
    #         'IoU sky',
    #         'IoU void'
    #     ]
    # }

    # Préparer les données pour le tableau avec des noms COURTS
    table_data = {
        'Métrique': [
            'IoU M.',      # Au lieu de 'IoU Modèle'
            'IoU R.',      # Au lieu de 'IoU Réel'
            'IoU P.',      # Au lieu de 'IoU Pondéré'
            'ms',          # Au lieu de 'Temps (ms)'
            'MB',          # Au lieu de 'Taille (MB)'
            '---',         # Séparateur
            'flat',        # Au lieu de 'IoU flat'
            'hum',         # Au lieu de 'IoU human'
            'veh',         # Au lieu de 'IoU vehicle'
            'cons',        # Au lieu de 'IoU construction'
            'obj',         # Au lieu de 'IoU object'
            'nat',         # Au lieu de 'IoU nature'
            'sky',         # Au lieu de 'IoU sky'
            'void'         # Au lieu de 'IoU void'
        ]
    }
    
    # Pour chaque modèle dans les résultats
    for model_key, result in comparison_results.items():
        project = "P8" if "P8" in model_key else "P9"
        model_name = model_key.replace("P8_", "").replace("P9_", "")
        
        # Pour les noms de colonnes, utiliser des versions courtes
        # Au lieu de "VGG16-UNet", utiliser "VGG16"
        # Au lieu de "SegFormer-B0 FP32", utiliser "SF-B0"
        short_name = model_name
        if "VGG16" in model_name:
            short_name = "VGG16"
        elif "SegFormer-B0" in model_name:
            short_name = "SF-B0" if "INT8" not in model_name else "SF-B0i"
        elif "SegFormer-B1" in model_name:
            short_name = "SF-B1" if "INT8" not in model_name else "SF-B1i"
        elif "SegFormer-B2" in model_name:
            short_name = "SF-B2" if "INT8" not in model_name else "SF-B2i"

        # Récupérer la config du modèle
        model_config = MODELS_CONFIG[project].get(model_name, {})
        
        # Colonnes du tableau
        col_values = []
        
        # IoU Modèle (depuis config)
        col_values.append(f"{model_config.get('iou', 0):.3f}")
        
        # IoU Réel et Pondéré (calculés si GT disponible)
        iou_real = "-"
        iou_weighted = "-"
        iou_per_class = {}
        
        if mask_gt is not None:
            # Calculer l'IoU réel si on a le masque prédit
            pred_mask_raw = None
            
            # Extraire le masque prédit
            mask_keys = ['mask', 'colored_mask', 'segmentation_image', 'segmentation_base64']
            for key in mask_keys:
                if key in result:
                    mask_img = base64_to_image(result[key])
                    pred_mask_raw = np.array(mask_img)
                    break
            
            if pred_mask_raw is not None:
                # Si c'est un masque RGB, le convertir en indices
                if len(pred_mask_raw.shape) == 3:
                    pred_mask_raw = rgb_to_class_indices(pred_mask_raw)
                
                # Calculer les métriques
                iou_metrics = calculate_iou_metrics(pred_mask_raw, mask_gt, debug=False)
                iou_real = f"{iou_metrics['iou_global']:.3f}"
                iou_weighted = f"{iou_metrics['iou_weighted']:.3f}"
                
                # IoU par classe avec logique correcte
                for class_id in range(8):
                    # Compter les pixels dans GT et prédiction
                    pixels_gt = iou_metrics['pixels_per_class'].get(class_id, 0)
                    pred_mask_flat = pred_mask_raw.flatten()
                    pixels_pred = np.sum(pred_mask_flat == class_id)
                    
                    # Logique de décision
                    if pixels_gt == 0 and pixels_pred == 0:
                        # Classe absente du GT ET de la prédiction
                        # C'est correct, on met NaN (sera affiché en gris/neutre)
                        iou_per_class[class_id] = np.nan
                    else:
                        # Classe présente dans GT ou prédiction ou les deux
                        # On utilise la valeur IoU calculée (peut être 0.0)
                        iou_value = iou_metrics['iou_per_class'].get(class_id, 0.0)
                        iou_per_class[class_id] = iou_value
        else:
            # Pas de GT disponible
            for class_id in range(8):
                iou_per_class[class_id] = np.nan
        
        col_values.append(iou_real)
        col_values.append(iou_weighted)
        
        # Temps d'inférence
        time_keys = ['inference_time', 'processing_time_ms', 'inference_time_ms']
        inference_time = 0
        for key in time_keys:
            if key in result:
                inference_time = result[key]
                break
        col_values.append(f"{inference_time:.0f}")
        
        # Taille du modèle
        col_values.append(f"{model_config.get('size_mb', 0):.1f}")
        
        # Séparateur
        col_values.append("---")
        
        # IoU par classe avec formatage approprié
        for class_id in range(8):
            if class_id in iou_per_class:
                value = iou_per_class[class_id]
                if pd.isna(value):
                    col_values.append(np.nan)  # Garder NaN pour les classes absentes partout
                else:
                    col_values.append(f"{value:.3f}")
            else:
                col_values.append(np.nan)
        
        # Ajouter la colonne au tableau
        table_data[model_name] = col_values
    
    return pd.DataFrame(table_data)



###### /ajout IoU

def create_metrics_comparison(p8_result, p9_results):
    """Crée un tableau comparatif des métriques"""
    data = []
    
    # Ajouter P8
    if p8_result:
        data.append({
            'Projet': 'P8',
            'Modèle': 'VGG16-UNet',
            'IoU': MODELS_CONFIG['P8']['VGG16-UNet']['iou'],
            'Temps (ms)': p8_result.get('inference_time', 0),
            'Taille (MB)': MODELS_CONFIG['P8']['VGG16-UNet']['size_mb'],
            'Technique': MODELS_CONFIG['P8']['VGG16-UNet']['technique']
        })
    
    # Ajouter P9
    for model_name, result in p9_results.items():
        if result:
            data.append({
                'Projet': 'P9',
                'Modèle': model_name,
                'IoU': MODELS_CONFIG['P9'][model_name]['iou'],
                'Temps (ms)': result.get('inference_time', 0),
                'Taille (MB)': MODELS_CONFIG['P9'][model_name]['size_mb'],
                'Technique': MODELS_CONFIG['P9'][model_name]['technique']
            })
    
    return pd.DataFrame(data)
    """Crée un tableau comparatif des métriques"""
    data = []
    
    # Ajouter P8
    if p8_result:
        data.append({
            'Projet': 'P8',
            'Modèle': 'VGG16-UNet',
            'IoU': MODELS_CONFIG['P8']['VGG16-UNet']['iou'],
            'Temps (ms)': p8_result.get('inference_time', 0),
            'Taille (MB)': MODELS_CONFIG['P8']['VGG16-UNet']['size_mb'],
            'Technique': MODELS_CONFIG['P8']['VGG16-UNet']['technique']
        })
    
    # Ajouter P9
    for model_name, result in p9_results.items():
        if result:
            data.append({
                'Projet': 'P9',
                'Modèle': model_name,
                'IoU': MODELS_CONFIG['P9'][model_name]['iou'],
                'Temps (ms)': result.get('inference_time', 0),
                'Taille (MB)': MODELS_CONFIG['P9'][model_name]['size_mb'],
                'Technique': MODELS_CONFIG['P9'][model_name]['technique']
            })
    
    return pd.DataFrame(data)

def colorize_mask(mask_array):
    """Applique la palette de couleurs Cityscapes à un masque en niveaux de gris"""
    # Palette Cityscapes CORRIGÉE
    CITYSCAPES_PALETTE = np.array([
        [128, 64, 128],   # 0: flat (route) - violet
        [220, 20, 60],    # 1: human (personne) - rouge
        [0, 0, 142],      # 2: vehicle - bleu foncé
        [70, 70, 70],     # 3: construction - gris foncé (CORRIGÉ!)
        [153, 153, 153],  # 4: object - gris clair
        [107, 142, 35],   # 5: nature - vert
        [70, 130, 180],   # 6: sky - bleu ciel
        [0, 0, 0]         # 7: void - noir
    ])
    
    # Convertir le masque en array numpy si ce n'est pas déjà fait
    if isinstance(mask_array, Image.Image):
        mask_array = np.array(mask_array)
    
    # Créer l'image colorée
    h, w = mask_array.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Appliquer la couleur pour chaque classe
    for class_id in range(8):
        mask = (mask_array == class_id)
        colored_mask[mask] = CITYSCAPES_PALETTE[class_id]
    
    return Image.fromarray(colored_mask)

def get_overlay_from_result(result):
    """Extrait ou crée l'overlay depuis un résultat"""
    if 'overlay' in result:
        return base64_to_image(result['overlay'])
    elif 'overlay_image' in result:
        return base64_to_image(result['overlay_image'])
    else:
        # Créer l'overlay
        mask_img = get_mask_from_result(result)
        if mask_img:
            return create_overlay(st.session_state.selected_image, mask_img, alpha=0.6)
    return None

def get_mask_from_result(result):
    """Extrait le masque coloré depuis un résultat"""
    mask_keys = ['colored_mask', 'mask', 'segmentation_image', 'segmentation_base64']
    for key in mask_keys:
        if key in result:
            mask_img = base64_to_image(result[key])
            mask_np = np.array(mask_img)
            if len(mask_np.shape) == 2 or (len(mask_np.shape) == 3 and mask_np.shape[2] == 1):
                return colorize_mask(mask_np)
            return mask_img
    return None

# Interface principale
def main():
    # Initialisation session state
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    

    # Charger l'index des images
    index_data = load_image_index()
    
    def compact_divider():
        st.markdown("<hr style='margin: 0.5rem 0; border: 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)


    # Sidebar avec tous les contrôles
    
    with st.sidebar:
        st.container()
        
        # État des APIs
        # st.markdown("### 🔌 État des APIs")
        # st.markdown("<div style='margin-top: -4rem; margin-bottom: -4rem;'>### 🔌 État des APIs</div>", unsafe_allow_html=True)        
        
        st.markdown("""
            <div style='margin-top: -7rem;'>
                <h1 style='margin-bottom: 0.5rem;'>🚗 Comparaison des modèles de segmentation</h1>
            </div>
        """, unsafe_allow_html=True)
        # compact_divider()
        st.markdown("""
            <div style='margin-top: -2rem;'>
                <h3 style='margin-bottom: 0.5rem;'>🔌 État des APIs</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            p8_ok, p8_health = check_api_health('P8')
            if p8_ok:
                st.success("✓ P8 OK")
            else:
                st.error("✗ P8 KO")
        
        with col2:
            p9_ok, p9_health = check_api_health('P9')
            if p9_ok:
                st.success("✓ P9 OK")
            else:
                st.error("✗ P9 KO")
        
        # st.markdown("---")
        compact_divider()
        
        # Sélection des modèles
        st.markdown("### 🎯 Sélection des modèles")
        
        # Checkbox P8 seul
        use_p8 = st.checkbox("VGG16-UNet (P8)", value=True, disabled=not p8_ok)
        
        # Espace de séparation et titre P9
        st.markdown("<div style='margin-top: 1rem; margin-bottom: 1.5rem;'><b>Modèles P9 :</b></div>", unsafe_allow_html=True)
        # st.markdown("<div style='margin-top: 0.0rem; margin-bottom: 1.5rem;'><b>Modèles P9 :</b></div>", unsafe_allow_html=True)

        # Checkboxes P9 sur 2 colonnes
        p9_models_list = list(MODELS_CONFIG['P9'].keys())
        col1, col2 = st.columns(2)
        
        selected_p9_models = []
        for idx, model_name in enumerate(p9_models_list):
            if idx % 2 == 0:
                with col1:
                    if st.checkbox(model_name.replace("SegFormer-", "SF-"), 
                                key=f"p9_{model_name}",
                                value=model_name == 'SegFormer-B1 FP32', 
                                disabled=not p9_ok):
                        selected_p9_models.append(model_name)
            else:
                with col2:
                    if st.checkbox(model_name.replace("SegFormer-", "SF-"), 
                                key=f"p9_{model_name}",
                                value=model_name == 'SegFormer-B1 FP32', 
                                disabled=not p9_ok):
                        selected_p9_models.append(model_name)
        
        # st.markdown("---")
        compact_divider()
        
        # Sélection de l'image
        st.markdown("### 🖼️ Sélection de l'image")

        select_tab, upload_tab = st.tabs(["Exemples", "Upload"])

        # Initialiser les variables de session si nécessaire
        if 'pending_image' not in st.session_state:
            st.session_state.pending_image = None
        if 'pending_image_filename' not in st.session_state:
            st.session_state.pending_image_filename = None
        if 'pending_image_name' not in st.session_state:
            st.session_state.pending_image_name = None
                
        with select_tab:
            if index_data['images']:
                image_names = [img['name'] for img in index_data['images']]
                selected_name = st.selectbox("Choisir un exemple", image_names, key="image_selector")
                
                if selected_name:
                    selected_img = next(img for img in index_data['images'] if img['name'] == selected_name)
                    image_path = IMAGES_PATH / selected_img['filename']
                    if image_path.exists():
                        # Stocker dans session_state pour persister entre les reruns
                        st.session_state.pending_image = load_image(image_path)
                        st.session_state.pending_image_filename = selected_img['filename']
                        st.session_state.pending_image_name = selected_name
                        
                        # Afficher l'aperçu
                        st.image(st.session_state.pending_image, caption=f"Aperçu: {selected_name}", use_column_width=True)
                        
                        # # Message différent selon si c'est une nouvelle image ou pas
                        # if (hasattr(st.session_state, 'selected_image_filename') and 
                        #     st.session_state.selected_image_filename == st.session_state.pending_image_filename):
                        #     st.success("✅ Image actuelle - Vous pouvez changer les modèles et relancer")
                        # else:
                        #     st.info("💡 Nouvelle image - Cliquez sur 'Lancer la comparaison' pour l'utiliser")

        with upload_tab:
            uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                # Stocker dans session_state
                st.session_state.pending_image = Image.open(uploaded_file)
                st.session_state.pending_image_filename = uploaded_file.name
                st.session_state.pending_image_name = "Image uploadée"
                
                st.image(st.session_state.pending_image, caption="Aperçu: Image uploadée", use_column_width=True)
                # st.info("💡 Cette image sera utilisée lors du clic sur 'Lancer la comparaison'")

        # st.markdown("---")
        compact_divider()

        # BOUTON DE LANCEMENT
        if st.session_state.pending_image is not None:
            # Vérifier qu'au moins un modèle est sélectionné
            if not use_p8 and not selected_p9_models:
                st.warning("⚠️ Sélectionnez au moins un modèle")
                st.button("🚀 Lancer la comparaison", type="primary", use_container_width=True, disabled=True)
            else:
                # Déterminer le texte du bouton selon le contexte
                button_text = "🚀 Lancer la comparaison"
                if (hasattr(st.session_state, 'selected_image_filename') and 
                    st.session_state.selected_image_filename == st.session_state.pending_image_filename and
                    hasattr(st.session_state, 'comparison_results') and 
                    st.session_state.comparison_results):
                    button_text = "🔄 Relancer la comparaison"
                
                if st.button(button_text, type="primary", use_container_width=True):
                    # Transférer l'image pending vers selected
                    st.session_state.selected_image = st.session_state.pending_image
                    st.session_state.selected_image_filename = st.session_state.pending_image_filename
                    st.session_state.launch_comparison = True
                    # Note: pas besoin de st.rerun() ici, Streamlit va automatiquement rerun après le clic
        else:
            st.info("📸 Sélectionnez d'abord une image")
            st.button("🚀 Lancer la comparaison", type="primary", use_container_width=True, disabled=True)

        # st.markdown("---")
        compact_divider()
        
        # Légende des classes sur 2 colonnes
        st.markdown("### 🎨 Classes Cityscapes")
        col1, col2 = st.columns(2)
        
        with col1:
            for label, color_rgb in list(COLOR_MAPPING.items())[:4]:
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin: 4px 0;">'
                    f'<div style="width: 16px; height: 16px; background-color: {color_rgb}; '
                    f'border-radius: 3px; margin-right: 8px;"></div>'
                    f'<span style="font-size: 13px;">{label}</span></div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            for label, color_rgb in list(COLOR_MAPPING.items())[4:]:
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin: 4px 0;">'
                    f'<div style="width: 16px; height: 16px; background-color: {color_rgb}; '
                    f'border-radius: 3px; margin-right: 8px;"></div>'
                    f'<span style="font-size: 13px;">{label}</span></div>',
                    unsafe_allow_html=True
                )
        
        # # # NOUVEAU : Afficher le statut de comparaison ici
        # try:
        #     if results:
        #         if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results:
        #             # compact_divider()
        #             st.success(f"✅ Comparaison terminée - {len(st.session_state.comparison_results)} modèles")
        #         # else:
        #         #     st.error("❌ Aucun résultat obtenu")
        # except:
        #     print("except")
        

    # Zone principale - Plus d'espace pour les résultats
    if hasattr(st.session_state, 'launch_comparison') and st.session_state.launch_comparison:
        # Exécuter la comparaison
        # [Code de comparaison ici...]
        
        # Convertir l'image
        buffered = io.BytesIO()
        st.session_state.selected_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode()
        
        # Progress bar
        progress = st.progress(0)
        status = st.empty()
        
        results = {}
        total_models = (1 if use_p8 else 0) + len(selected_p9_models)
        current = 0
        
        # Prédiction P8
        if use_p8:
            status.text("🔮 Segmentation VGG16-UNet (P8)...")
            start_time = time.time()
            success, result = predict_p8(image_base64)
            inference_time = (time.time() - start_time) * 1000
            
            if success:
                result['inference_time'] = inference_time
                results['P8_VGG16-UNet'] = result
            current += 1
            progress.progress(current / total_models)
        
        # Prédictions P9
        for model_name in selected_p9_models:
            status.text(f"🔮 Segmentation {model_name}...")
            start_time = time.time()
            
            buffered.seek(0)  # Reset buffer
            success, result = predict_p9(buffered, model_name)
            inference_time = (time.time() - start_time) * 1000
            
            if success:
                result['inference_time'] = inference_time
                results[f'P9_{model_name}'] = result
            current += 1
            progress.progress(current / total_models)
        
        status.empty()
        progress.empty()
        
        if results:
            st.session_state.comparison_results = results
        #     # st.success(f"✅ Comparaison terminée - {len(results)} modèles")
        else:
            st.error("❌ Aucun résultat obtenu")
        # Réinitialiser le flag
        st.session_state.launch_comparison = False
        
    # Affichage des résultats
    if st.session_state.comparison_results:
        # st.markdown("---")
        # compact_divider()
        # st.header("📊 Résultats de la comparaison")
        
        # Tabs pour l'organisation
        tab1, tab2 = st.tabs(["🖼️ Visualisations", "📈 Métriques"])
        
        with tab1:  # Onglet Visualisations
            if st.session_state.comparison_results:
                
                if st.session_state.selected_image:
                    # Chercher le masque ground truth
                    mask_gt = None
                    mask_gt_colored = None
                    overlay_gt = None
                    
                    if hasattr(st.session_state, 'selected_image_filename'):
                        base_name = st.session_state.selected_image_filename.replace('.png', '')
                        mask_filename = f"{base_name}_mask.png"
                        mask_path = IMAGES_PATH / mask_filename
                        
                        if mask_path.exists():
                            mask_gt = load_image(mask_path)
                            mask_gt_colored = colorize_mask(mask_gt)
                            overlay_gt = create_overlay(st.session_state.selected_image, mask_gt_colored, alpha=0.6)
                            
                        # Affichage de la donnée de référence
                        if mask_gt_colored and overlay_gt:
                            # Container avec titre vertical et images
                            col_title, col_images = st.columns([0.2, 11.5])  # Colonne étroite pour le titre
                            
                            with col_title:
                                st.markdown("""
                                <div style="
                                    height: 200px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                ">
                                    <div style="
                                        transform: rotate(270deg);
                                        transform-origin: center;
                                        white-space: nowrap;
                                        font-size: 16px;
                                        font-weight: bold;
                                        color: #333;
                                    ">
                                        Images de référence 📷
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_images:
                                col1, col2, col3 = st.columns([1, 1, 1])
                                
                                with col1:
                                    st.image(st.session_state.selected_image, 
                                            caption="Image originale", 
                                            use_column_width=True)
                                
                                with col2:
                                    st.image(mask_gt_colored,
                                            caption="Masque ground truth", 
                                            use_column_width=True)
                                
                                with col3:
                                    st.image(overlay_gt,
                                            caption="Overlay GT", 
                                            use_column_width=True)                        


                # Préparer les options pour les deux sélecteurs
                model_options = list(st.session_state.comparison_results.keys())
                model_display_names = [m.replace("P8_", "").replace("P9_", "") for m in model_options]
                
                # Options pour le sélecteur de gauche (avec Original et GT si disponible)
                left_options = ["Image originale"]
                left_keys = ["original"]
                
                if mask_gt_colored:  # Si on a un ground truth
                    left_options.append("Ground Truth")
                    left_keys.append("ground_truth")
                
                # Ajouter tous les modèles
                left_options.extend(model_display_names)
                left_keys.extend(model_options)
                
                # Options pour le sélecteur de droite (modèles uniquement)
                right_options = model_display_names
                right_keys = model_options
                
                # Colonnes pour swipe et tableauc
                col_swipe, col_tableau = st.columns([3, 5])
                
                with col_swipe:
                    # Deux colonnes pour les sélecteurs
                    col_left_select, col_right_select = st.columns(2)
                    
                    with col_left_select:
                        left_idx = st.selectbox(
                            "🖼️ Image de gauche :",
                            range(len(left_options)),
                            format_func=lambda x: f"◀ {left_options[x]}",  # Flèche gauche                            
                            key="select_left_comparison",
                            label_visibility="collapsed"
                        )
                        selected_left_key = left_keys[left_idx]

                    with col_right_select:
                         # Déterminer l'index par défaut intelligent
                        default_right_idx = 0

                        # Si gauche = VGG16-UNet et c'est le premier à droite
                        if (selected_left_key in right_keys and 
                            right_keys[0] == selected_left_key and 
                            len(right_keys) > 1):
                            default_right_idx = 1  # Passer au modèle suivant

                        right_idx = st.selectbox(
                            "hidden_label",  # Pas de label
                            range(len(right_options)),
                            index=default_right_idx,  # Index intelligent
                            format_func=lambda x: f"{right_options[x]} ▶",  # Flèche droite
                            key="select_right_comparison",
                            label_visibility="collapsed"
                        )
                        selected_right_key = right_keys[right_idx]
                    

                    # Préparer les images selon les sélections
                    left_image = None
                    left_label = left_options[left_idx]
                    
                    # Image de gauche
                    if selected_left_key == "original":
                        left_image = st.session_state.selected_image
                    elif selected_left_key == "ground_truth":
                        left_image = overlay_gt if overlay_gt else mask_gt_colored
                    else:
                        # C'est un modèle
                        result = st.session_state.comparison_results[selected_left_key]
                        left_image = get_overlay_from_result(result)  # Fonction helper à créer
                    
                    # Image de droite (toujours un modèle)
                    right_result = st.session_state.comparison_results[selected_right_key]
                    right_image = get_overlay_from_result(right_result)
                    right_label = right_options[right_idx]
                

                # Afficher le swipe unique
                if left_image and right_image:
                    # Info sur la comparaison
                    # st.info(f"Comparaison : **{left_label}** ↔️ **{right_label}**")
                    
                    # # Colonnes pour swipe et tableauc
                    # # col_swipe, col_tableau = st.columns([3, 4])
                    # col_swipe, col_tableau = st.columns([3, 5])
                    with col_swipe:
                        # Le swipe
                        image_comparison(
                            img1=left_image,
                            img2=right_image,
                            label1=left_label,
                            label2=right_label,
                            width=540,
                            starting_position=50,
                            show_labels=True,
                            make_responsive=True,
                            in_memory=True
                        )
                        
                        # # Afficher le masque en dessous
                        mask_colored = get_mask_from_result(right_result)
                        show_mask = st.checkbox("🎭 Afficher le masque de segmentation", value=True)
                        if show_mask and mask_colored:
                            st.image(mask_colored, caption=f"Masque {right_label}", use_column_width=True)

                    with col_tableau:
                        st.markdown("##### 📊 Métriques")
                        
                        # Créer un tableau pour tout les modeles selectionnes
                        
                        comparison_df = create_comparative_iou_table(
                            st.session_state.comparison_results, 
                            mask_gt if 'mask_gt' in locals() else None
                        )
                        
                        # Afficher le tableau
                        # Fonction pour styler les cellules selon les valeurs IoU
                        def style_iou_cell(val):
                            """
                            Colorie les cellules selon la valeur IoU
                            - NaN/vide : gris neutre (classe absente partout)
                            - 0.000 : rouge (vraie erreur)
                            - >0 : code couleur selon performance
                            """
                            # Convertir en string pour gérer tous les cas
                            val_str = str(val).strip()
                            
                            # Cas spéciaux pour les non-valeurs et séparateurs
                            if val_str in ["-", "---"]:
                                return ""  # Pas de couleur pour les séparateurs
                            
                            # NaN = classe absente du GT et de la prédiction (cas neutre)
                            if val_str in ["nan", "NaN", "None", ""]:
                                return "background-color: #F0F0F0; color: #999999"  # Gris clair neutre
                            
                            try:
                                value = float(val_str)
                                
                                # Appliquer le code couleur normal (y compris pour 0.0)
                                if value >= 0.8:
                                    return 'background-color: #90EE90'  # Vert clair - Excellent
                                elif value >= 0.6:
                                    return 'background-color: #98FB98'  # Vert pâle - Bon
                                elif value >= 0.4:
                                    return 'background-color: #FFD700'  # Or - Moyen
                                elif value >= 0.2:
                                    return 'background-color: #FFE4B5'  # Pêche - Faible
                                else:
                                    # Inclut 0.0 : vraie erreur (faux positif ou mauvaise segmentation)
                                    return 'background-color: #FFB6C1'  # Rouge clair - Très faible
                                    
                            except (ValueError, TypeError):
                                return ""  # Pas de couleur si conversion impossible
                        
                        # Appliquer le style au DataFrame
                        df_transposed = comparison_df.set_index('Métrique').T


                        # Identifier les colonnes IoU pour le styling
                        iou_columns = ['IoU Modèle', 'IoU Réel', 'IoU Pondéré', 
                                    'IoU flat', 'IoU human', 'IoU vehicle', 'IoU construction',
                                    'IoU object', 'IoU nature', 'IoU sky', 'IoU void']
                        
                        # Partie affichage avec légende améliorée

                        # Afficher le tableau
                        styled_df = comparison_df.set_index('Métrique').T

                        # Définir les colonnes à GARDER (sans IoU P.)
                        columns_to_keep = ['IoU M.', 'IoU R.', 'ms', 'MB', '---', 'flat', 'hum', 'veh', 'cons', 'obj', 'nat', 'sky', 'void']

                        # Filtrer seulement les colonnes qu'on veut
                        columns_to_display = [col for col in columns_to_keep if col in styled_df.columns]
                        styled_df = styled_df[columns_to_display]


                        # IMPORTANT : S'assurer que les colonnes numériques sont bien typées
                        # numeric_columns = ['IoU Modèle', 'IoU Réel', 'IoU Pondéré', 'Temps (ms)', 'Taille (MB)',
                        #                 'IoU flat', 'IoU human', 'IoU vehicle', 'IoU construction', 
                        #                 'IoU object', 'IoU nature', 'IoU sky', 'IoU void']
                        
                        numeric_columns = ['IoU M.', 'IoU R.', 'IoU P.', 'ms', 'MB', 'flat', 'hum', 'veh', 'cons', 'obj', 'nat', 'sky', 'void' ]

                        for col in numeric_columns:
                            if col in styled_df.columns:
                                styled_df[col] = pd.to_numeric(styled_df[col], errors='coerce')


                        # Appliquer le style
                        style_func = styled_df.style.applymap(style_iou_cell)

                        # Formater les NaN pour affichage
                        
                        style_func = style_func.format(na_rep='NaN', precision=3)

                        # st.dataframe(style_func, use_container_width=True)

                        # Gérer l'affichage selon l'espace disponible
                        try:
                            # Essayer d'afficher le tableau complet stylé
                            st.dataframe(style_func, use_container_width=True)
                            
                        except Exception as e:
                            # Si erreur (probablement largeur insuffisante)
                            st.info("📱 Mode compact activé")
                            
                            # Afficher seulement les colonnes essentielles
                            essential_cols = ['IoU M.', 'IoU R.', 'ms', 'MB']
                            if '---' in styled_df.columns:
                                # Exclure le séparateur et les colonnes après
                                idx_separator = list(styled_df.columns).index('---')
                                essential_df = styled_df.iloc[:, :idx_separator]
                            else:
                                essential_df = styled_df[essential_cols]
                            
                            # Afficher version compacte
                            st.dataframe(essential_df, use_container_width=True)
                            
                            # Option pour voir les détails
                            if st.button("📊 Voir IoU par classe"):
                                class_cols = ['flat', 'hum', 'veh', 'cons', 'obj', 'nat', 'sky', 'void']
                                class_df = styled_df[[col for col in class_cols if col in styled_df.columns]]
                                st.dataframe(class_df, use_container_width=True)


                        # Légende
                        with st.expander("📖 Légende des couleurs"):
                            st.markdown("""**Codes couleur IoU :**
                                - 🟢 Vert clair : IoU ≥ 0.80 (Excellent)
                                - 🟢 Vert : IoU ≥ 0.60 (Bon)
                                - 🟡 Or : IoU ≥ 0.40 (Moyen)
                                - 🟠 Pêche : IoU ≥ 0.20 (Faible)
                                - 🔴 Rouge clair : IoU < 0.20 (Très faible)
                                - ⬜ Gris : Classe absente (NaN)""")
                                                    

                        # Métriques comparatives uniquement si on compare deux modèles
                        if (selected_left_key not in ["original", "ground_truth"] and 
                            selected_right_key in st.session_state.comparison_results):
                            
                            # Récupérer les configs et résultats
                            left_result = st.session_state.comparison_results[selected_left_key]
                            right_result = st.session_state.comparison_results[selected_right_key]
                            
                            left_model_name = selected_left_key.replace("P8_", "").replace("P9_", "")
                            right_model_name = selected_right_key.replace("P8_", "").replace("P9_", "")
                            
                            # Configs pour IoU théorique
                            left_project = "P8" if "P8" in selected_left_key else "P9"
                            right_project = "P8" if "P8" in selected_right_key else "P9"
                            left_config = MODELS_CONFIG[left_project].get(left_model_name, {})
                            right_config = MODELS_CONFIG[right_project].get(right_model_name, {})
                            
                            # IoU théoriques
                            left_iou_model = left_config.get('iou', 0)
                            right_iou_model = right_config.get('iou', 0)
                            
                            # NOUVEAU : Calculer les IoU réels si on a le GT
                            left_iou_real = None
                            right_iou_real = None
                            
                            if mask_gt is not None:
                                # Calculer IoU réel pour le modèle de gauche
                                left_mask = get_mask_from_result(left_result)
                                if left_mask:
                                    left_mask_np = np.array(left_mask)
                                    if len(left_mask_np.shape) == 3:
                                        left_mask_np = rgb_to_class_indices(left_mask_np)
                                    left_metrics = calculate_iou_metrics(left_mask_np, mask_gt, debug=False)
                                    left_iou_real = left_metrics['iou_global']
                                
                                # Calculer IoU réel pour le modèle de droite
                                right_mask = get_mask_from_result(right_result)
                                if right_mask:
                                    right_mask_np = np.array(right_mask)
                                    if len(right_mask_np.shape) == 3:
                                        right_mask_np = rgb_to_class_indices(right_mask_np)
                                    right_metrics = calculate_iou_metrics(right_mask_np, mask_gt, debug=False)
                                    right_iou_real = right_metrics['iou_global']
                            
                            # Tailles et temps
                            left_size = left_config.get('size_mb', 0)
                            right_size = right_config.get('size_mb', 0)
                            size_diff = right_size - left_size
                            
                            left_time = left_result.get('inference_time', 0)
                            right_time = right_result.get('inference_time', 0)
                            time_diff = right_time - left_time
                            
                            # Afficher les métriques
                            st.markdown(f"**{left_model_name} → {right_model_name}**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Utiliser IoU réel si disponible, sinon IoU modèle
                                if left_iou_real is not None and right_iou_real is not None:
                                    iou_diff = right_iou_real - left_iou_real
                                    delta_color = "normal" if iou_diff >= 0 else "inverse"
                                    st.metric(
                                        label="Δ IoU (réel)",
                                        value=f"{right_iou_real:.3f}",
                                        delta=f"{iou_diff:+.3f}",
                                        delta_color=delta_color,
                                        help=f"IoU réel mesuré sur cette image\nGauche: {left_iou_real:.3f} | Droite: {right_iou_real:.3f}"
                                    )
                                else:
                                    # Fallback sur IoU théorique
                                    iou_diff = right_iou_model - left_iou_model
                                    delta_color = "normal" if iou_diff >= 0 else "inverse"
                                    st.metric(
                                        label="Δ IoU (modèle)",
                                        value=f"{right_iou_model:.3f}",
                                        delta=f"{iou_diff:+.3f}",
                                        delta_color=delta_color,
                                        help=f"IoU théorique du modèle\nGauche: {left_iou_model:.3f} | Droite: {right_iou_model:.3f}"
                                    )
                            
                            with col2:
                                delta_color = "inverse" if time_diff >= 0 else "normal"
                                st.metric(
                                    label="Δ Temps",
                                    value=f"{right_time:.0f} ms",
                                    delta=f"{time_diff:+.0f} ms",
                                    delta_color=delta_color
                                )
                            
                            with col3:
                                delta_color = "inverse" if size_diff >= 0 else "normal"
                                st.metric(
                                    label="Δ Taille",
                                    value=f"{right_size:.1f} MB",
                                    delta=f"{size_diff:+.1f} MB",
                                    delta_color=delta_color
                                )

        with tab2:
            # Préparer les données pour le DataFrame
            p8_result = st.session_state.comparison_results.get('P8_VGG16-UNet')
            p9_results = {k.replace('P9_', ''): v for k, v in st.session_state.comparison_results.items() if 'P9' in k}
            
            df = create_metrics_comparison(p8_result, p9_results)
            
            if not df.empty:
                # Graphique de comparaison
                st.subheader("Performance vs Efficacité")
                chart = create_comparison_chart(df)
                st.altair_chart(chart, use_container_width=True)
                
                # Tableau des métriques
                st.subheader("Tableau comparatif")
                
                # Formater le DataFrame pour l'affichage
                styled_df = df.style.format({
                    'IoU': '{:.3f}',
                    'Temps (ms)': '{:.0f}',
                    'Taille (MB)': '{:.1f}'
                })
                
                # Colorier les meilleures valeurs
                def highlight_best(s):
                    if s.name == 'IoU':
                        return ['background-color: lightgreen' if v == s.max() else '' for v in s]
                    elif s.name in ['Temps (ms)', 'Taille (MB)']:
                        return ['background-color: lightgreen' if v == s.min() else '' for v in s]
                    return ['' for _ in s]
                
                styled_df = styled_df.apply(highlight_best)
                st.dataframe(styled_df)
        

if __name__ == "__main__":
    main()
