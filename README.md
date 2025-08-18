# 🚀 Projet 9 - Segmentation Sémantique Améliorée avec SegFormer

## 📊 Démonstration en ligne

🌐 **Dashboard comparatif** : https://oc-p9-dashboard.azurewebsites.net

## 🎯 Objectif

Amélioration des performances de segmentation sémantique d'images urbaines par l'implémentation de **SegFormer** (Vision Transformer, NeurIPS 2021) par rapport à la baseline VGG16-UNet du Projet 8.

## 📈 Résultats clés

| Métrique | Baseline P8 | SegFormer P9 | Amélioration |
|----------|-------------|--------------|--------------|
| **IoU moyen** | 63.1% | 77.0% | **+22.2%** |
| **Temps inférence** | 8.5s | 0.43s | **-95%** |
| **Taille modèle** | 98.8 MB | 14.2 MB | **-85.6%** |

## 🏗️ Structure du projet

```
├── api_segformer/      # API REST avec 6 modèles SegFormer
├── dashboard/          # Interface Streamlit de comparaison
├── notebooks/          # Notebooks d'entraînement et évaluation
├── docs/              # Documentation et livrables
└── README.md
```

## 🚀 Infrastructure déployée sur Azure

| Service | URL | Statut |
|---------|-----|--------|
| **API Baseline P8** | https://oc-p8-segmentation.azurewebsites.net | ✅ Opérationnel |
| **API SegFormer P9** | https://oc-p9-segformer.azurewebsites.net | ✅ Opérationnel |
| **Dashboard** | https://oc-p9-dashboard.azurewebsites.net | ✅ Opérationnel |

## 💻 Installation locale

### API SegFormer
```bash
cd api_segformer
make setup
make run-b1-int8
```

### Dashboard
```bash
cd dashboard
make setup
make run
```

## 📚 Documentation

- **[Plan de travail](docs/DRACHE_Didier_1_plan_travail_072025.pdf)** : Planification et méthodologie
- **[Note méthodologique](docs/DRACHE_Didier_3_note_methodo_072025.pdf)** : Analyse détaillée et résultats
- **[Notebook d'entraînement](notebooks/P9_segformer_entrainement_v19.ipynb)** : Entraînement des modèles
- **[Notebook de comparaison](notebooks/comparaison_09.ipynb)** : Évaluation comparative

## 🔬 Modèles disponibles

6 variantes déployées : SegFormer B0/B1/B2 × FP32/INT8

| Modèle | Taille | IoU | Temps Azure |
|--------|--------|-----|-------------|
| B1 INT8 (défaut) | 14.2 MB | 74.8% | ~430ms |
| B2 FP32 (meilleur) | 104.9 MB | 77.0% | ~980ms |

## 🛠️ Technologies

- **Deep Learning** : PyTorch, Transformers, ONNX Runtime
- **API** : Flask, Flask-CORS
- **Dashboard** : Streamlit
- **Cloud** : Azure App Service F1
- **Dataset** : Cityscapes (8 classes)

## 👤 Auteur

**Didier DRACHE** - Data Scientist  
Projet 9 - OpenClassrooms

---
*Projet réalisé dans le cadre du parcours Data Scientist d'OpenClassrooms*
