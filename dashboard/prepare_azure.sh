#!/bin/bash
echo "📦 Préparation du dashboard pour Azure..."

# Créer le fichier startup
echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > startup.txt

# Créer un fichier de configuration pour Azure
cat > config_azure.py << CONFIG
# Configuration des APIs pour Azure
API_URLS = {
    'baseline': 'https://oc-p8-segmentation.azurewebsites.net',
    'segformer': 'https://oc-p9-segformer.azurewebsites.net'
}
CONFIG

echo "✅ Dashboard prêt pour Azure"
