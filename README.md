# oc-p9-segmentation-amelioree

## 🚀 Structure du Projet

### API SegFormer (`/api_segformer`)
API REST pour la segmentation sémantique avec 6 modèles SegFormer (B0/B1/B2 × FP32/INT8).

```bash
cd api_segformer
make setup
make run-b1-int8  # Lancer avec le modèle B1 INT8
make test         # Tester l'API

URL de production : https://oc-p9-segformer.azurewebsites.net