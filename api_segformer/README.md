# API SegFormer - Segmentation Sémantique P9

API REST pour la segmentation sémantique d'images urbaines utilisant les modèles SegFormer (B0, B1, B2).

## 🚀 Déploiement Azure

URL de production : https://oc-p9-segformer.azurewebsites.net

## 📦 Modèles Disponibles

| Modèle | Type | Taille | IoU | Temps (CPU) |
|--------|------|--------|-----|-------------|
| SegFormer-B0 | FP32 | 14MB | 69.8% | 83ms |
| SegFormer-B0 | INT8 | 4.3MB | 69.5% | 42ms |
| SegFormer-B1 | FP32 | 53MB | 75.2% | 156ms |
| SegFormer-B1 | INT8 | 14MB | 74.8% | 78ms |
| SegFormer-B2 | FP32 | 105MB | 77.0% | 312ms |
| SegFormer-B2 | INT8 | 28MB | 76.5% | 156ms |

## 🛠️ Installation Locale

```bash
# Créer l'environnement virtuel
make setup

# Lancer l'API (différents modèles)
make run-b0-int8    # B0 quantifié (recommandé pour test rapide)
make run-b1-int8    # B1 quantifié (meilleur compromis)
make run-b2-fp32    # B2 full precision (meilleure qualité)

# Tester l'API
make test
```

## 🌐 Endpoints

- `GET /` : Status de l'API
- `POST /segment` : Segmentation d'image
  - Body: multipart/form-data avec image
  - Query params: `?model=segformer_b1_int8`

## ☁️ Déploiement Azure

```bash
# Préparer le package de déploiement
make azure-prepare

# Déployer sur Azure
make azure-deploy
```

## 📊 Performances

Testé sur Azure App Service F1 (1 CPU, 1GB RAM) :
- B0 INT8 : ~500ms par image
- B1 INT8 : ~800ms par image  
- B2 INT8 : ~1500ms par image
