# Makefile pour l'application Streamlit Comparative P8 vs P9

# Variables
VENV_NAME = venv_streamlit_p9
PYTHON = python3.10
VENV_PYTHON = $(VENV_NAME)/bin/python
VENV_PIP = $(VENV_NAME)/bin/pip

# Configuration APIs
API_URL_P8_LOCAL = http://localhost:5000
API_URL_P8_AZURE = https://oc-p8-segmentation.azurewebsites.net
API_URL_P9_LOCAL = http://localhost:8000
API_URL_P9_AZURE = https://oc-p9-segformer.azurewebsites.net
STREAMLIT_PORT = 8502

# Azure
AZURE_APP_NAME = oc-p9-dashboard
AZURE_RESOURCE_GROUP = rg-p9-segformer

# Informations système
SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help setup install run run-p8 run-p9 run-both test clean azure-deploy azure-prepare azure-update-env azure-restart azure-status azure-logs azure-test azure-destroy

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║        Dashboard Comparatif Segmentation P8 vs P9            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 DÉMARRAGE RAPIDE:"
	@echo "  make setup              - Installation complète"
	@echo "  make run-local          - Lancer avec APIs locales"
	@echo "  make run-azure          - Lancer avec APIs Azure"
	@echo "  make run-mixed          - API P8 Azure + API P9 locale"
	@echo ""
	@echo "📦 INSTALLATION:"
	@echo "  make venv               - Créer l'environnement virtuel"
	@echo "  make install            - Installer les dépendances"
	@echo "  make copy-images        - Copier les images de test"
	@echo ""
	@echo "🎯 MODES DE LANCEMENT:"
	@echo "  make run-local          - Tout en local (P8:5000, P9:8000)"
	@echo "  make run-azure          - Tout sur Azure"
	@echo "  make run-mixed          - Mode mixte (utile pour tests)"
	@echo "  make run-p8-only        - Tester uniquement P8"
	@echo "  make run-p9-only        - Tester uniquement P9"
	@echo ""
	@echo "☁  AZURE - DÉPLOIEMENT:"
	@echo "  make azure-deploy       - Déploiement complet sur Azure"
	@echo "  make azure-prepare      - Préparer le package de déploiement"
	@echo "  make azure-update       - Mise à jour rapide"
	@echo "  make azure-update-env   - MAJ variables d'environnement"
	@echo "  make azure-restart      - Redémarrer l'app Azure"
	@echo ""
	@echo "☁  AZURE - MONITORING:"
	@echo "  make azure-status       - Vérifier le statut"
	@echo "  make azure-health       - Check santé des 3 services"
	@echo "  make azure-logs         - Voir les logs en temps réel"
	@echo "  make azure-test         - Tester l'app déployée"
	@echo ""
	@echo "☁  AZURE - MAINTENANCE:"
	@echo "  make azure-stop         - Arrêter le dashboard (économie)"
	@echo "  make azure-start        - Redémarrer le dashboard"
	@echo "  make azure-stop-all     - Arrêter TOUS les services P8+P9"
	@echo "  make azure-start-all    - Redémarrer TOUS les services"
	@echo "  make azure-destroy      - Supprimer le dashboard"
	@echo ""
	@echo "🧪 TESTS:"
	@echo "  make test               - Tests de syntaxe et imports"
	@echo "  make test-apis          - Tester la connexion aux APIs"
	@echo "  make test-local         - Tester en local complet"
	@echo ""
	@echo "🧹 NETTOYAGE:"
	@echo "  make clean              - Nettoyer les fichiers temporaires"
	@echo "  make clean-all          - Tout nettoyer (incluant venv)"
	@echo ""
	@echo "📊 COMPARAISON:"
	@echo "  make compare-local      - Comparaison avec APIs locales"
	@echo "  make compare-azure      - Comparaison avec APIs Azure"
	@echo "  make benchmark          - Benchmark complet P8 vs P9"

# Installation complète
setup: venv install copy-images
	@echo "✅ Installation terminée!"
	@echo ""
	@echo "🚀 Pour démarrer:"
	@echo "  - APIs locales: make run-local"
	@echo "  - APIs Azure:   make run-azure"

# Créer le virtualenv
$(VENV_NAME):
	@echo "🔧 Création de l'environnement virtuel Python $(PYTHON)..."
	@which $(PYTHON) > /dev/null || (echo "❌ Erreur: $(PYTHON) n'est pas installé" && exit 1)
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "✅ Environnement virtuel créé"

venv: $(VENV_NAME)

# Installer les dépendances
install: $(VENV_NAME)
	@echo "📦 Installation des dépendances..."
	$(VENV_PIP) install --upgrade pip
	@echo "📚 Installation des packages Streamlit..."
	$(VENV_PIP) install streamlit==1.28.0
	$(VENV_PIP) install requests==2.31.0
	$(VENV_PIP) install Pillow==10.0.0
	$(VENV_PIP) install numpy==1.24.3
	$(VENV_PIP) install pandas==2.0.3
	$(VENV_PIP) install altair==5.0.1
	$(VENV_PIP) install streamlit-image-comparison==0.0.4
	@echo "✅ Dépendances installées"

# Copier les images de test
copy-images:
	@echo "📸 Copie des images de test..."
	@if [ -d "../projet8/app_web/test_images" ]; then \
		cp -r ../projet8/app_web/test_images .; \
		echo "✅ Images P8 copiées"; \
	elif [ -d "../p8/test_images" ]; then \
		cp -r ../p8/test_images .; \
		echo "✅ Images P8 copiées"; \
	else \
		echo "⚠  Images P8 non trouvées, création d'exemples..."; \
		mkdir -p test_images; \
		echo '{"images": [], "description": "Images de test"}' > test_images/index.json; \
	fi

# ================== LANCEMENT LOCAL ==================

# Lancer avec les deux APIs en local
run-local: $(VENV_NAME)
	@echo "🚀 Lancement du dashboard (APIs locales)..."
	@echo "📍 Dashboard: http://localhost:$(STREAMLIT_PORT)"
	@echo "🔗 API P8: $${API_URL_P8}"
	@echo "🔗 API P9: $${API_URL_P9}"
	@echo ""
	@echo "⚠  Assurez-vous que les APIs sont lancées:"
	@echo "  Terminal P8: cd ../projet8/api && make run-h5"
	@echo "  Terminal P9: cd ../projet9_api && make run-b1-fp32"
	@echo ""
	API_URL_P8=$(API_URL_P8) API_URL_P9=$(API_URL_P9) \
		$(VENV_PYTHON) -m streamlit run app_streamlit_comparative.py \
		--server.port $(STREAMLIT_PORT)

# Lancer avec les deux APIs sur Azure
run-azure: $(VENV_NAME)
	@echo "🚀 Lancement du dashboard (APIs Azure)..."
	@echo "📍 Dashboard: http://localhost:$(STREAMLIT_PORT)"
	@echo "☁  API P8: $(API_URL_P8_AZURE)"
	@echo "☁  API P9: $(API_URL_P9_AZURE)"
	API_URL_P8=$(API_URL_P8_AZURE) API_URL_P9=$(API_URL_P9_AZURE) \
		$(VENV_PYTHON) -m streamlit run app_streamlit_comparative.py \
		--server.port $(STREAMLIT_PORT)

# Mode mixte: P8 Azure + P9 local
run-mixed: $(VENV_NAME)
	@echo "🚀 Lancement du dashboard (mode mixte)..."
	@echo "📍 Dashboard: http://localhost:$(STREAMLIT_PORT)"
	@echo "☁  API P8: $(API_URL_P8_AZURE)"
	@echo "🔗 API P9: $(API_URL_P9_LOCAL) (local)"
	API_URL_P8=$(API_URL_P8_AZURE) API_URL_P9=$(API_URL_P9_LOCAL) \
		$(VENV_PYTHON) -m streamlit run app_streamlit_comparative.py \
		--server.port $(STREAMLIT_PORT)

# Tester uniquement P8
run-p8-only: $(VENV_NAME)
	@echo "🚀 Test P8 uniquement..."
	API_URL_P8=$(API_URL_P8_AZURE) API_URL_P9="" \
		$(VENV_PYTHON) -m streamlit run app_streamlit_comparative.py \
		--server.port $(STREAMLIT_PORT)

# Tester uniquement P9
run-p9-only: $(VENV_NAME)
	@echo "🚀 Test P9 uniquement..."
	API_URL_P8="" API_URL_P9=$(API_URL_P9_AZURE) \
		$(VENV_PYTHON) -m streamlit run app_streamlit_comparative.py \
		--server.port $(STREAMLIT_PORT)

# ================== TESTS ==================

# Tests basiques
test: $(VENV_NAME)
	@echo "🧪 Test de l'application..."
	@echo "1. Vérification de la syntaxe..."
	@$(VENV_PYTHON) -m py_compile app_streamlit_comparative.py
	@echo "✅ Syntaxe OK"
	@echo ""
	@echo "2. Vérification des imports..."
	@$(VENV_PYTHON) -c "import streamlit; import requests; import pandas; import altair; print('✅ Imports OK')"

# Tester la connexion aux APIs
test-apis: $(VENV_NAME)
	@echo "🔍 Test de connexion aux APIs..."
	@echo ""
	@echo "📡 Test API P8 (Azure):"
	@curl -s -w "   Status: %{http_code} - Temps: %{time_total}s\n" \
		$(API_URL_P8_AZURE)/health -o /dev/null || echo "   ❌ P8 non accessible"
	@echo ""
	@echo "📡 Test API P9 (Azure):"
	@curl -s -w "   Status: %{http_code} - Temps: %{time_total}s\n" \
		$(API_URL_P9_AZURE)/health -o /dev/null || echo "   ❌ P9 non accessible"

# Test local complet
test-local: test test-apis
	@echo ""
	@echo "✅ Tests locaux terminés"

# ================== AZURE DEPLOYMENT ==================

# Préparation du package Azure
azure-prepare:
	@echo "📦 Préparation du package Azure..."
	@mkdir -p azure_deploy
	@mkdir -p azure_deploy/.streamlit
	@cp app.py azure_deploy/app.py
	@echo "📝 Création du requirements.txt..."
	@echo "streamlit==1.28.0" > azure_deploy/requirements.txt
	@echo "requests==2.31.0" >> azure_deploy/requirements.txt
	@echo "Pillow==10.0.0" >> azure_deploy/requirements.txt
	@echo "numpy==1.24.3" >> azure_deploy/requirements.txt
	@echo "pandas==2.0.3" >> azure_deploy/requirements.txt
	@echo "altair==5.0.1" >> azure_deploy/requirements.txt
	@echo "streamlit-image-comparison==0.0.4" >> azure_deploy/requirements.txt
	@echo "matplotlib==3.7.2" >> azure_deploy/requirements.txt
	@cp -r test_images azure_deploy/ 2>/dev/null || echo "⚠  Images non trouvées"
	@echo "🔧 Configuration Streamlit pour Azure..."
	@echo "[server]" > azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "port = 8000" >> azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "enableCORS = false" >> azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "enableXsrfProtection = true" >> azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "[browser]" >> azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "serverAddress = \"0.0.0.0\"" >> azure_deploy/.streamlit/config.toml
	@echo "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0" > azure_deploy/startup.txt
	@echo "✅ Package Azure prêt dans azure_deploy/"

# Déploiement complet Azure
azure-deploy: azure-prepare
	@echo "☁  Déploiement du dashboard comparatif sur Azure..."
	@echo "⚠  Note: Le dashboard nécessite que les APIs P8 et P9 soient déployées"
	
	# Étape 1: Déployer l'application
	@echo "📤 Déploiement de l'application..."
	cd azure_deploy && az webapp up \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--sku F1 \
		--runtime "python|3.10"
	
	# Étape 2: Configurer le démarrage
	@echo "⚙️  Configuration du démarrage Streamlit..."
	az webapp config set \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--startup-file "python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0"
	
	# Étape 3: Configurer les variables d'environnement
	@echo "🔧 Configuration des APIs..."
	az webapp config appsettings set \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--settings \
		API_URL_P8="$(API_URL_P8_AZURE)" \
		API_URL_P9="$(API_URL_P9_AZURE)" \
		AZURE_DEPLOYMENT="true"
	
	# Étape 4: Redémarrer
	@echo "♻️  Redémarrage de l'application..."
	az webapp restart \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP)
	
	@echo ""
	@echo "✅ Dashboard comparatif déployé!"
	@echo "🌐 URL: https://$(AZURE_APP_NAME).azurewebsites.net"
	@echo ""
	@echo "📊 Architecture complète:"
	@echo "  Dashboard: https://$(AZURE_APP_NAME).azurewebsites.net"
	@echo "  API P8:    $(API_URL_P8_AZURE)"
	@echo "  API P9:    $(API_URL_P9_AZURE)"
	@echo ""
	@echo "⏱️  Le dashboard sera disponible dans ~60 secondes"

# Mise à jour rapide
azure-update: azure-prepare
	@echo "🔄 Mise à jour rapide du dashboard..."
	cd azure_deploy && zip -r ../app.zip . -q
	az webapp deployment source config-zip \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--src app.zip
	rm app.zip
	@echo "✅ Dashboard mis à jour"

# Mise à jour des variables d'environnement
azure-update-env:
	@echo "🔧 Mise à jour des variables d'environnement..."
	az webapp config appsettings set \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--settings \
		API_URL_P8="$(API_URL_P8_AZURE)" \
		API_URL_P9="$(API_URL_P9_AZURE)"
	@echo "✅ Variables mises à jour"
	@make azure-restart

# Redémarrer l'application
azure-restart:
	@echo "♻️  Redémarrage du dashboard..."
	az webapp restart \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP)
	@echo "✅ Dashboard redémarré"

# ================== MONITORING AZURE ==================

# Status du dashboard
azure-status:
	@echo "📊 Statut du Dashboard Comparatif:"
	@echo ""
	@echo "🔍 État de l'application:"
	@az webapp show \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--query "{État:state, URL:defaultHostName}" -o table 2>/dev/null || echo "❌ Dashboard non trouvé"
	@echo ""
	@echo "⚙️  Variables configurées:"
	@az webapp config appsettings list \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--query "[?contains(name, 'API_URL')].{API:name, URL:value}" \
		-o table 2>/dev/null || echo "Pas de variables"

# Check santé des 3 services
azure-health:
	@echo "🏥 Vérification santé des 3 services:"
	@echo ""
	@echo "1️⃣  API P8 (VGG16-UNet):"
	@curl -s -w "   Status: %{http_code} - Temps: %{time_total}s\n" \
		$(API_URL_P8_AZURE)/health -o /dev/null || echo "   ❌ Non accessible"
	@echo ""
	@echo "2️⃣  API P9 (SegFormer):"
	@curl -s -w "   Status: %{http_code} - Temps: %{time_total}s\n" \
		$(API_URL_P9_AZURE)/health -o /dev/null || echo "   ❌ Non accessible"
	@echo ""
	@echo "3️⃣  Dashboard Comparatif:"
	@curl -s -w "   Status: %{http_code} - Temps: %{time_total}s\n" \
		https://$(AZURE_APP_NAME).azurewebsites.net -o /dev/null || echo "   ❌ Non accessible"

# Logs en temps réel
azure-logs:
	@echo "📜 Logs du dashboard (Ctrl+C pour arrêter)..."
	az webapp log tail \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP)

# Test du dashboard déployé
azure-test:
	@echo "🧪 Test du dashboard déployé..."
	@echo ""
	@echo "1️⃣  Test de connectivité:"
	@curl -s -o /dev/null -w "   HTTP Status: %{http_code} - Temps: %{time_total}s\n" \
		https://$(AZURE_APP_NAME).azurewebsites.net || echo "   ❌ Erreur"
	@echo ""
	@echo "2️⃣  Pour test complet:"
	@echo "   🌐 Ouvrez: https://$(AZURE_APP_NAME).azurewebsites.net"
	@echo "   📸 Uploadez une image"
	@echo "   ✅ Sélectionnez P8 et P9"
	@echo "   🚀 Lancez la comparaison"

# ================== GESTION DES CRÉDITS AZURE ==================

# Arrêter le dashboard (économie)
azure-stop:
	@echo "⏸️  Arrêt du dashboard..."
	az webapp stop \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP)
	@echo "✅ Dashboard arrêté (économie de crédits)"

# Redémarrer le dashboard
azure-start:
	@echo "▶️  Redémarrage du dashboard..."
	az webapp start \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP)
	@echo "✅ Dashboard redémarré"

# Arrêter TOUS les services (P8 + P9 + Dashboard)
azure-stop-all:
	@echo "⏸️  Arrêt de TOUS les services P8 + P9..."
	@echo "  🔸 Arrêt API P8..."
	-@az webapp stop --name oc-p8-segmentation --resource-group rg-p8-segmentation 2>/dev/null
	@echo "  🔸 Arrêt API P9..."
	-@az webapp stop --name oc-p9-segformer --resource-group $(AZURE_RESOURCE_GROUP) 2>/dev/null
	@echo "  🔸 Arrêt Dashboard..."
	-@az webapp stop --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP) 2>/dev/null
	@echo ""
	@echo "✅ Tous les services sont arrêtés (économie maximale)"

# Redémarrer TOUS les services
azure-start-all:
	@echo "▶️  Redémarrage de TOUS les services..."
	@echo "  🔸 Démarrage API P8..."
	-@az webapp start --name oc-p8-segmentation --resource-group rg-p8-segmentation 2>/dev/null
	@echo "  🔸 Démarrage API P9..."
	-@az webapp start --name oc-p9-segformer --resource-group $(AZURE_RESOURCE_GROUP) 2>/dev/null
	@echo "  🔸 Démarrage Dashboard..."
	-@az webapp start --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP) 2>/dev/null
	@echo ""
	@echo "✅ Tous les services redémarrés"
	@echo "⏱️  Attendre ~60 secondes pour la disponibilité complète"

# Supprimer le dashboard
azure-destroy:
	@echo "🗑️  Suppression du dashboard Azure..."
	@echo "App: $(AZURE_APP_NAME)"
	@read -p "Confirmer la suppression? (oui/non): " confirm; \
	if [ "$$confirm" = "oui" ]; then \
		az webapp delete \
			--name $(AZURE_APP_NAME) \
			--resource-group $(AZURE_RESOURCE_GROUP) \
			--keep-empty-plan; \
		echo "✅ Dashboard supprimé"; \
	else \
		echo "❌ Suppression annulée"; \
	fi

# ================== COMPARAISON & BENCHMARK ==================

# Comparaison avec APIs locales
compare-local:
	@echo "📊 Lancement de la comparaison (APIs locales)..."
	@make run-local

# Comparaison avec APIs Azure
compare-azure:
	@echo "📊 Lancement de la comparaison (APIs Azure)..."
	@make run-azure

# Benchmark complet
benchmark: test-apis
	@echo "⚡ Benchmark P8 vs P9..."
	@echo ""
	@echo "📊 Métriques théoriques:"
	@echo "┌─────────────────┬──────────┬──────────┬──────────┐"
	@echo "│ Modèle          │ IoU      │ Taille   │ Technique│"
	@echo "├─────────────────┼──────────┼──────────┼──────────┤"
	@echo "│ VGG16-UNet (P8) │ 0.631    │ 98.8 MB  │ CNN      │"
	@echo "│ SegFormer-B0    │ 0.698    │ 14.5 MB  │ ViT      │"
	@echo "│ SegFormer-B1    │ 0.701    │ 52.5 MB  │ ViT      │"
	@echo "│ SegFormer-B2    │ 0.760    │ 104.9 MB │ ViT      │"
	@echo "└─────────────────┴──────────┴──────────┴──────────┘"
	@echo ""
	@echo "Pour benchmark temps réel: make run-azure"

# ================== NETTOYAGE ==================

# Nettoyage basique
clean:
	@echo "🧹 Nettoyage..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .streamlit/
	rm -rf azure_deploy/
	rm -f app.zip
	@echo "✅ Nettoyage terminé"

# Nettoyage complet
clean-all: clean
	@echo "🗑️  Suppression de l'environnement virtuel..."
	rm -rf $(VENV_NAME)
	rm -rf test_images/
	@echo "✅ Nettoyage complet terminé"

# ================== WORKFLOW SOUTENANCE ==================

# Workflow pour la soutenance
demo-prepare:
	@echo "🎯 Préparation de la démo pour soutenance..."
	@echo ""
	@echo "1️⃣  Installation locale..."
	@make setup
	@echo ""
	@echo "2️⃣  Vérification des APIs..."
	@make test-apis
	@echo ""
	@echo "3️⃣  Démarrage des services Azure..."
	@make azure-start-all
	@echo ""
	@echo "✅ Système prêt pour la démonstration!"
	@echo ""
	@echo "📊 URLs pour la soutenance:"
	@echo "  P8 API:    $(API_URL_P8_AZURE)"
	@echo "  P9 API:    $(API_URL_P9_AZURE)"
	@echo "  Dashboard: https://$(AZURE_APP_NAME).azurewebsites.net"

# Démo complète
demo-run:
	@echo "🚀 Lancement de la démo complète..."
	@make azure-start-all
	@sleep 60
	@make azure-health
	@echo ""
	@echo "🌐 Ouvrir: https://$(AZURE_APP_NAME).azurewebsites.net"

# Alias pour compatibilité
deploy: azure-deploy
status: azure-status
logs: azure-logs

.DEFAULT_GOAL := help