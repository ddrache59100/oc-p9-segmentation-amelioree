# src/config_azure.py - Configuration minimale pour Azure F1
import os

class Config:
    # Dimensions
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    NUM_CLASSES = 8
    
    # Classes Cityscapes
    CITYSCAPES_CLASSES = ['flat', 'human', 'vehicle', 'construction', 
                          'object', 'nature', 'sky', 'void']
      
    # Couleurs pour visualisation (RGB) - PALETTE CORRIGÉE
    CITYSCAPES_COLORS = [
        [128, 64, 128],   # flat - violet
        [220, 20, 60],    # human - rouge
        [0, 0, 142],      # vehicle - bleu foncé
        [70, 70, 70],     # construction - gris foncé (CORRIGÉ de 190,153,153)
        [153, 153, 153],  # object - gris clair
        [107, 142, 35],   # nature - vert
        [70, 130, 180],   # sky - bleu ciel
        [0, 0, 0]         # void - noir
    ]

    MODELS = {
    # B0
    'segformer_b0_int8': {
        'path': 'models/segformer_b0/model_quantized_azure.onnx',  # Version corrigée
        'feature_extractor': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
        'size_mb': 4.6,
        'iou': 0.587,
        'precision': 'INT8'
    },
    # B1
    'segformer_b1_int8': {
        'path': 'models/segformer_b1/model_quantized_azure.onnx',  # Version corrigée
        'feature_extractor': 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
        'size_mb': 14.2,
        'iou': 0.667,
        'precision': 'INT8'
    },
    # ... etc
}
    
    # Modèle par défaut pour Azure
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'segformer_b1_fp32')
    
    # API Settings
    PORT = int(os.environ.get('PORT', 8000))
    DEBUG = False
    