# tests/compare_results.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

def load_images(image_name):
    """Charge l'image originale, le masque GT et la prédiction"""
    base_name = image_name.replace('.png', '')
    
    # Chemins
    original_path = f"test_images/{base_name}.png"
    gt_path = f"test_images/{base_name}_mask.png"
    pred_path = f"test_results/{base_name}_segmented.png"
    
    # Charger les images
    original = Image.open(original_path) if os.path.exists(original_path) else None
    gt = Image.open(gt_path) if os.path.exists(gt_path) else None
    pred = Image.open(pred_path) if os.path.exists(pred_path) else None
    
    return original, gt, pred

def calculate_iou(gt, pred):
    """Calcule l'IoU approximatif entre GT et prédiction"""
    if gt is None or pred is None:
        return 0
    
    # Convertir en arrays numpy
    gt_array = np.array(gt)
    pred_array = np.array(pred)
    
    # Si les images sont en couleur, les convertir en labels
    # (approximation basique pour la démo)
    if len(gt_array.shape) == 3:
        # Simplification : on prend juste le canal rouge pour identifier les classes
        gt_array = gt_array[:, :, 0] // 32  # Diviser en 8 classes approximatives
    if len(pred_array.shape) == 3:
        pred_array = pred_array[:, :, 0] // 32
    
    # Calculer l'IoU global (approximatif)
    intersection = np.sum(gt_array == pred_array)
    total = gt_array.size
    iou = intersection / total if total > 0 else 0
    
    return iou

def visualize_comparison(image_name="test_000.png"):
    """Visualise la comparaison entre original, GT et prédiction"""
    
    original, gt, pred = load_images(image_name)
    
    if pred is None:
        print(f"❌ Pas de prédiction trouvée pour {image_name}")
        print("   Lancez d'abord: make test-quick")
        return
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3 if gt else 2, figsize=(15, 5))
    
    # Image originale
    if original:
        axes[0].imshow(original)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
    
    # Ground Truth (si disponible)
    if gt:
        axes[1].imshow(gt)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prédiction
        axes[2].imshow(pred)
        axes[2].set_title('Prédiction SegFormer')
        axes[2].axis('off')
        
        # Calculer et afficher l'IoU
        iou = calculate_iou(gt, pred)
        fig.suptitle(f'Comparaison - Accuracy approximative: {iou*100:.1f}%', fontsize=14)
    else:
        # Prédiction seule
        axes[1].imshow(pred)
        axes[1].set_title('Prédiction SegFormer')
        axes[1].axis('off')
        fig.suptitle('Résultat de Segmentation', fontsize=14)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = f"test_results/comparison_{image_name}"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✅ Comparaison sauvée: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    image_name = sys.argv[1] if len(sys.argv) > 1 else "test_000.png"
    visualize_comparison(image_name)