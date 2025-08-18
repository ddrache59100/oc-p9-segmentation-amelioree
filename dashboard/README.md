# Dashboard Comparatif - Segmentation Sémantique

Interface Streamlit pour comparer les performances de segmentation entre :
- **Baseline P8** : VGG16-UNet
- **Nouvelle méthode P9** : SegFormer

## 🚀 Lancement local

```bash
make setup
make run
```

## ☁️ Déploiement Azure

```bash
make azure-deploy
```

## 🌐 URLs de production

- Dashboard : https://oc-p9-dashboard.azurewebsites.net
- API P8 : https://oc-p8-segmentation.azurewebsites.net
- API P9 : https://oc-p9-segformer.azurewebsites.net

## 📊 Fonctionnalités

- Upload d'images personnalisées
- Sélection d'images de test pré-chargées
- Comparaison côte à côte des résultats
- Métriques de performance (IoU, temps)
- Visualisation des distributions de classes
