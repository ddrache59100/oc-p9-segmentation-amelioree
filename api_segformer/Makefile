# Makefile pour l'API SegFormer P9

# Variables
VENV_NAME = venv_p9
PYTHON = python3.10
VENV_PYTHON = $(VENV_NAME)/bin/python
VENV_PIP = $(VENV_NAME)/bin/pip

# Variables pour les tests
TEST_IMAGE = test_images/test_000.png
TEST_IMAGES = test_images/test_00*.png
API_URL ?= http://localhost:5000

# Variables Azure
AZURE_APP_NAME ?= oc-p9-segformer
AZURE_RESOURCE_GROUP ?= rg-p9-segformer

.PHONY: help venv install run test clean all

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║          API SegFormer P9 - Commandes disponibles             ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 DÉMARRAGE RAPIDE:"
	@echo "  make setup        - Installation complète (venv + dépendances)"
	@echo "  make dev          - Mode développement (B1 INT8 par défaut)"
	@echo "  make test-quick   - Test rapide avec une image"
	@echo ""
	@echo "📦 INSTALLATION:"
	@echo "  make venv         - Créer l'environnement virtuel"
	@echo "  make install      - Installer les dépendances"
	@echo ""
	@echo "🏃 EXÉCUTION - MODÈLES INT8:"
	@echo "  make run          - Lancer l'API (B1 INT8 par défaut)"
	@echo "  make run-b0       - SegFormer-B0 INT8 (4.6 MB, IoU: 0.587)"
	@echo "  make run-b1       - SegFormer-B1 INT8 (14.2 MB, IoU: 0.667)"
	@echo "  make run-b2       - SegFormer-B2 INT8 (28.3 MB, IoU: 0.705)"
	@echo ""
	@echo "🏃 EXÉCUTION - MODÈLES FP32:"
	@echo "  make run-b0-fp32  - SegFormer-B0 FP32 (14.5 MB, IoU: 0.698)"
	@echo "  make run-b1-fp32  - SegFormer-B1 FP32 (52.5 MB, IoU: 0.701)"
	@echo "  make run-b2-fp32  - SegFormer-B2 FP32 (104.9 MB, IoU: 0.760)"
	@echo ""
	@echo "🧪 TESTS:"
	@echo "  make test         - Tests unitaires de l'API"
	@echo "  make test-quick   - Test rapide avec une image"
	@echo "  make test-all     - Tests sur toutes les images"
	@echo "  make test-batch   - Test batch sur 5 premières images"
	@echo "  make test-visual  - Test avec visualisation"
	@echo ""
	@echo "⚡ BENCHMARKS:"
	@echo "  make benchmark       - Benchmark des 3 modèles INT8"
	@echo "  make benchmark-full  - Benchmark complet INT8 vs FP32"
	@echo "  make benchmark-detail - Benchmark détaillé avec statistiques"
	@echo ""
	@echo "☁️  AZURE - DÉPLOIEMENT:"
	@echo "  make azure-prepare    - Préparer le package de déploiement"
	@echo "  make azure-check      - Vérifier le package avant déploiement"
	@echo "  make azure-deploy     - Déployer sur Azure F1"
	@echo "  make azure-update     - Mettre à jour l'app déployée"
	@echo ""
	@echo "☁️  AZURE - MONITORING:"
	@echo "  make azure-test       - Tester l'API déployée"
	@echo "  make azure-status     - Vérifier le statut Azure"
	@echo "  make azure-logs       - Voir les logs en temps réel"
	@echo "  make azure-stop       - Arrêter l'app (économiser crédits)"
	@echo "  make azure-start      - Redémarrer l'app"
	@echo ""
	@echo "🧹 MAINTENANCE:"
	@echo "  make clean        - Nettoyer les fichiers temporaires"
	@echo "  make clean-all    - Nettoyer tout (y compris venv)"
	@echo "  make status       - Voir l'état complet du projet"
	@echo "  make refresh      - Recréer l'environnement (clean + setup)"
	@echo ""
	@echo "📊 VISUALISATION:"
	@echo "  make visualize    - Visualiser les résultats de segmentation"

# Installation complète
setup: venv install
	@echo "✅ Installation complète terminée!"
	@echo "Lancez 'make dev' pour démarrer"

# Créer le virtualenv
$(VENV_NAME):
	@echo "🔧 Création de l'environnement virtuel..."
	@which $(PYTHON) > /dev/null || (echo "❌ Erreur: $(PYTHON) n'est pas installé" && exit 1)
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "✅ Environnement virtuel créé"

venv: $(VENV_NAME)

# Installer les dépendances
install: $(VENV_NAME)
	@echo "📦 Installation des dépendances..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "✅ Dépendances installées"

# Mode développement
dev: $(VENV_NAME)
	@echo "🚀 Lancement en mode développement..."
	@echo "📍 URL: http://localhost:5000"
	@echo "🤖 Modèle: SegFormer-B1 INT8 (14.2 MB, IoU: 0.667)"
	@echo "💡 Mode Lazy Loading activé - le modèle sera chargé à la première requête"
	@echo ""
	$(VENV_PYTHON) src/app.py

# Lancer avec différents modèles INT8
run: dev

run-b0: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B0 INT8 (4.6 MB)..."
	DEFAULT_MODEL=segformer_b0_int8 $(VENV_PYTHON) src/app.py

run-b1: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B1 INT8 (14.2 MB)..."
	DEFAULT_MODEL=segformer_b1_int8 $(VENV_PYTHON) src/app.py

run-b2: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B2 INT8 (28.3 MB)..."
	DEFAULT_MODEL=segformer_b2_int8 $(VENV_PYTHON) src/app.py

# Lancer les modèles FP32
run-b0-fp32: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B0 FP32 (14.5 MB)..."
	DEFAULT_MODEL=segformer_b0_fp32 $(VENV_PYTHON) src/app.py

run-b1-fp32: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B1 FP32 (52.5 MB)..."
	DEFAULT_MODEL=segformer_b1_fp32 $(VENV_PYTHON) src/app.py

run-b2-fp32: $(VENV_NAME)
	@echo "🚀 Lancement avec SegFormer-B2 FP32 (104.9 MB)..."
	DEFAULT_MODEL=segformer_b2_fp32 $(VENV_PYTHON) src/app.py

# Tests
test: $(VENV_NAME)
	@echo "🧪 Test de l'API..."
	$(VENV_PYTHON) tests/test_api.py

test-quick: $(VENV_NAME)
	@echo "🧪 Test rapide avec une image..."
	@echo "Image: $(TEST_IMAGE)"
	API_URL=$(API_URL) $(VENV_PYTHON) tests/test_api.py $(TEST_IMAGE)

test-all: $(VENV_NAME)
	@echo "🧪 Test avec toutes les images..."
	@for img in $(TEST_IMAGES); do \
		echo "Testing $$img..."; \
		API_URL=$(API_URL) $(VENV_PYTHON) tests/test_api.py $$img; \
	done

test-batch: $(VENV_NAME)
	@echo "🧪 Test batch sur plusieurs images..."
	@for i in 000 001 002 003 004; do \
		echo "\n📸 Test image test_$$i.png..."; \
		$(VENV_PYTHON) tests/test_api.py test_images/test_$$i.png; \
	done
	@echo "\n✅ Batch test terminé"
	@echo "Résultats dans: test_results/"

# Visualisation des résultats
visualize: $(VENV_NAME)
	@echo "📊 Visualisation des résultats..."
	@$(VENV_PYTHON) tests/compare_results.py

test-visual: test-quick visualize
	@echo "✅ Test et visualisation terminés"

# Benchmark
benchmark: $(VENV_NAME)
	@echo "⚡ Benchmark des 3 modèles INT8..."
	@echo "Assurez-vous que l'API est lancée!"
	@for model in b0 b1 b2; do \
		echo "\n📊 Test SegFormer-$$model INT8..."; \
		curl -X POST -F "image=@$(TEST_IMAGE)" \
			$(API_URL)/predict?model=segformer_$${model}_int8 \
			-w "\nTemps total: %{time_total}s\n"; \
	done

benchmark-full: $(VENV_NAME)
	@echo "⚡ Benchmark complet INT8 vs FP32..."
	@echo "Assurez-vous que l'API est lancée!"
	@echo ""
	@for variant in b0 b1 b2; do \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "📊 SegFormer-$$variant"; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "\n▶ FP32:"; \
		curl -s -X POST -F "image=@$(TEST_IMAGE)" \
			$(API_URL)/predict?model=segformer_$${variant}_fp32 \
			-w "\nTemps total: %{time_total}s\n" 2>/dev/null | grep -E "(model|inference_time)"; \
		echo "\n▶ INT8:"; \
		curl -s -X POST -F "image=@$(TEST_IMAGE)" \
			$(API_URL)/predict?model=segformer_$${variant}_int8 \
			-w "\nTemps total: %{time_total}s\n" 2>/dev/null | grep -E "(model|inference_time)"; \
		echo ""; \
	done

benchmark-detail: $(VENV_NAME)
	@echo "📊 Benchmark détaillé FP32 vs INT8..."
	@echo "Assurez-vous que l'API est lancée!"
	$(VENV_PIP) install tabulate 2>/dev/null || true
	$(VENV_PYTHON) tests/benchmark_complete.py

# Nettoyage
clean:
	@echo "🧹 Nettoyage..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf test_results/*.png
	rm -rf azure_deploy_minimal
	@echo "✅ Nettoyage terminé"

clean-all: clean
	@echo "🗑️  Suppression de l'environnement virtuel..."
	rm -rf $(VENV_NAME)
	@echo "✅ Nettoyage complet terminé"

refresh: clean-all setup
	@echo "♻️  Environnement recréé avec succès"

# Status
status:
	@echo "📊 État du projet:"
	@echo ""
	@echo "📁 Structure:"
	@echo "  - Modèles: $$(ls models/ | wc -l) variants"
	@echo "  - Images test: $$(ls test_images/*.png | grep -v mask | wc -l) images"
	@echo "  - Résultats: $$(ls test_results/*.png 2>/dev/null | wc -l) fichiers"
	@echo ""
	@echo "🤖 Modèles disponibles:"
	@ls -lh models/*/model_quantized.onnx 2>/dev/null || echo "  Modèles INT8 non trouvés"
	@ls -lh models/*/model.onnx 2>/dev/null || echo "  Modèles FP32 non trouvés"
	@echo ""
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "🐍 Environnement Python: ✅ Installé"; \
		$(VENV_PYTHON) --version; \
	else \
		echo "🐍 Environnement Python: ❌ Non installé (lancez 'make setup')"; \
	fi
	@echo ""
	@if [ -d "azure_deploy_minimal" ]; then \
		echo "☁️  Package Azure: ✅ Prêt ($$(du -sh azure_deploy_minimal | cut -f1))"; \
	else \
		echo "☁️  Package Azure: ❌ Non créé (lancez 'make azure-prepare')"; \
	fi

# ================== COMMANDES AZURE ==================

# Préparation Azure avec script adapté
azure-prepare:
	@echo "📦 Préparation du déploiement Azure..."
	@chmod +x prepare_minimal_deployment.sh
	./prepare_minimal_deployment.sh

azure-check:
	@echo "🔍 Vérification du package Azure..."
	@if [ -d "azure_deploy_minimal" ]; then \
		echo "📁 Contenu du package:"; \
		tree -h azure_deploy_minimal 2>/dev/null || ls -la azure_deploy_minimal/; \
		echo ""; \
		echo "📊 Taille totale: $$(du -sh azure_deploy_minimal | cut -f1)"; \
		echo "✅ Package prêt pour déploiement"; \
	else \
		echo "❌ Package non créé. Lancez 'make azure-prepare' d'abord"; \
	fi

# Déploiement Azure
azure-deploy: azure-prepare
	@echo "☁️  Déploiement sur Azure F1..."
	@echo "App: $(AZURE_APP_NAME)"
	@echo "Resource Group: $(AZURE_RESOURCE_GROUP)"
	cd azure_deploy_minimal && az webapp up \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--runtime "PYTHON:3.9" \
		--sku F1 \
		--location "West Europe"
	@echo ""
	@echo "✅ Déploiement terminé!"
	@echo "🌐 URL: https://$(AZURE_APP_NAME).azurewebsites.net"

azure-update: azure-prepare
	@echo "🔄 Mise à jour de l'application Azure..."
	cd azure_deploy_minimal && az webapp deployment source config-zip \
		--name $(AZURE_APP_NAME) \
		--resource-group $(AZURE_RESOURCE_GROUP) \
		--src @<(zip -r - .)
	@echo "✅ Application mise à jour"

# Tests Azure
azure-test: $(VENV_NAME)
	@echo "🧪 Test de l'API Azure..."
	@echo "URL: https://$(AZURE_APP_NAME).azurewebsites.net"
	API_URL=https://$(AZURE_APP_NAME).azurewebsites.net $(VENV_PYTHON) tests/test_api.py

# Monitoring Azure
azure-logs:
	@echo "📜 Logs Azure (Ctrl+C pour arrêter)..."
	az webapp log tail --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP)

azure-status:
	@echo "📊 Statut Azure:"
	@az webapp show --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP) \
		--query "{Status:state, URL:defaultHostName, Plan:appServicePlanId}" -o table 2>/dev/null || \
		echo "❌ App non trouvée"

azure-stop:
	@echo "⏸️  Arrêt de l'application Azure..."
	az webapp stop --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP)
	@echo "✅ Application arrêtée (économie de crédits)"

azure-start:
	@echo "▶️  Redémarrage de l'application Azure..."
	az webapp start --name $(AZURE_APP_NAME) --resource-group $(AZURE_RESOURCE_GROUP)
	@echo "✅ Application redémarrée"